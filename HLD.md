# SmolTorrent — High-Level Design

---

## What the system does

SmolTorrent distributes ML checkpoint files across a cluster of Raspberry Pi workers and reassembles them on demand. The Server is the master — it holds the original checkpoints, runs the API and watcher, and coordinates all operations. The Pis are dumb storage workers — they receive shards, hold them on disk, and serve them back on request. Every shard is replicated to a second worker (replication factor 2), so a single worker failure does not lose data.

```
  Training script / user
         │
         │  writes model.safetensors
         ▼
  ~/smolcluster/checkpoints/   (ckpt_root on Server)
         │
         │  watcher detects new file
         ▼
  FastAPI  (backend/api.py, port 8000)
    │   │   │   │
    ▼   ▼   ▼   ▼   TCP (Tailscale VPN, ~100 Mbps)
   Pi1 Pi2 Pi3 Pi4   (algorithms/SyncPS/worker.py)
  rank1 rank2 rank3 rank4

  Each Pi also runs:
    mDNS advertiser  (zeroconf _smoltorrent._tcp.local.)
    node_exporter    (Prometheus metrics port 9100)
    worker metrics   (port 920{rank})
```

---

## Why this topology

- **Master orchestrates, workers store** — the Server has MLX for tensor ops; Pis run pure torch. Keeping orchestration on the Server avoids cross-framework complexity on workers.
- **Tailscale VPN** — Pis are behind NAT; Tailscale gives every node a stable routable IP without port-forwarding or a public server.
- **Safetensors as wire format** — the only format that carries shape + dtype + tensor name across both MLX (Server) and torch (Pi) without importing the other framework.
- **Replication factor 2** — every shard is stored on two workers (round 0 + round 1). Gather falls back to the replica automatically if the primary is unreachable.
- **Zero-config discovery** — workers advertise over mDNS on startup; the master discovers them without any static IP configuration.

---

## Four user-facing operations

| Operation | Trigger | What happens |
|---|---|---|
| **Discover (grove)** | `grove start -n N` on master; `grove join` on each worker | Workers find master via mDNS TUI, POST to registration server, receive rank; master writes config.yaml and launches API + watcher |
| **Discover (SSH)** | `bash scripts/launch.sh` | rsync code to Pis, install deps, start API + watcher + workers in tmux |
| **Store** | `grove store --ckpt-path <path>` or watcher auto-detect | Split checkpoint → serialize shards once → push all N×REDUNDANCY sends in parallel |
| **Gather** | `grove gather --ckpt-path <path>` | Pull all shards in parallel (replica fallback per shard) → merge → `merged.safetensors` |
| **Watch** | `watcher/watch.py` daemon (always running) | Auto-detect new checkpoints, store them, crosscheck all workers |

---

## Components and their roles

| Component | Where it runs | Role |
|---|---|---|
| `backend/api.py` | Server, port 8000 | HTTP API — orchestrates store, gather, and discovery |
| `watcher/watch.py` | Server, daemon | Watches `ckpt_root`, triggers store automatically |
| `algorithms/SyncPS/worker.py` | Each Pi | TCP listener — stores and serves shards; advertises over mDNS; exposes Prometheus metrics |
| `networking/send_receive.py` | Both | Zero-copy TCP framing shared by master and workers |
| `networking/ssh_manager.py` | Server | Writes and updates the smoltorrent-managed block in `~/.ssh/config` |
| `discovery/__init__.py` | Server | Public API: `advertise_worker()`, `discover_workers()` |
| `discovery/grove/_mdns.py` | Both | zeroconf mDNS advertiser (`WorkerAdvertiser`) and scanner (`discover_mdns_workers`) |
| `discovery/grove/tui.py` | Server | Textual TUI — `WorkerPickerApp` for interactive node selection |
| `discovery/grove/transport/p2p.py` | Server (macOS) | AirDrop/AWDL peer discovery via Swift helper |
| `utils/common_utils.py` | Both | Tensor ops: chunk, serialize, deserialize, merge |
| `utils/shard_ops.py` | Server | HTTP client wrappers that call the API |
| `scripts/launch.sh` | Server (SSH setup) | rsync → deps → start all processes in tmux |
| `scripts/grove_launch.sh` | Server (grove flow) | Start API + watcher in tmux only; workers already up via `grove join` |
| `scripts/install_worker_service.sh` | Server (run once) | Install systemd auto-restart service on each Pi |
| `utils/boot_exporter.py` | All nodes | Prometheus exporter on port 9101 — exposes `smoltorrent_boot_time_ms` |

---

## Discover flow (high level)

### grove flow (no SSH)
```
grove start -n N                              // master
  → MasterAdvertiser: registers _smolt-master._tcp.local. via mDNS
  → HTTPServer on port 5999: waits for N worker POSTs

grove join                                    // each worker (TUI)
  → MasterBrowser: scans for _smolt-master._tcp.local.
  → JoinApp (Textual TUI): user selects master
  → POST {hostname, ip, user} → master:5999
  → master assigns rank, replies {rank, port}
  → worker starts algorithms/SyncPS/worker.py directly (no SSH)

  once N workers registered:
  → master writes configs/config.yaml (workers section + num_workers)
  → master runs grove_launch.sh: starts API + watcher in tmux
```

### SSH flow (production)
```
  → edit configs/config.yaml manually (hosts must match ~/.ssh/config aliases)
  → bash scripts/launch.sh
      rsyncs code to all Pis, installs deps, starts API + watcher + workers in tmux
```

---

## Store flow (high level)

```
Training writes checkpoint
  → watcher detects it
  → watcher calls POST /store-shard
  → API loads tensors, splits into N chunks, serializes each once (MLX not thread-safe)
  → builds N×REDUNDANCY jobs: shard i → workers[(i+round) % N] for round in 0,1
  → all 2N sends fire simultaneously in one ThreadPoolExecutor
  → each Pi writes shard.safetensors + shard.checksum to disk
  → failed sends → retry queue (daemon thread, exponential backoff, 6 retries max)
  → API streams "Done: 2N/2N sends (2x replicated)" → watcher crosschecks
```

## Gather flow (high level)

```
User runs: grove gather --ckpt-path <path>
  → calls POST /gather-shards
  → all N workers queried in parallel (ThreadPoolExecutor):
      → ("send_shard", rank, rel_path)
      ← shard_bytes
      if failed and REDUNDANCY > 1:
          immediately retry against workers[(i+1) % N]  ← replica fallback
      shard saved to disk on arrival (not buffered)
  → merge_shards([shards_by_index[i] for i in range(N)])
  → save → ckpt_root/{rel_path}/merged.safetensors
```

`shards_by_index` is keyed by shard index (0..N-1), not worker rank — so a replica serving shard 0 from rank 2 lands in the correct merge slot regardless of which worker actually served it.

## Watcher loop (high level)

On each trigger (new file or startup):
1. **file_sync** — ask all workers what they have → take intersection
2. **checksum_sync** — startup only: hash every existing shard on each Pi, verify against `.checksum` sidecar
3. **transfer** — push files missing from the intersection via `/store-shard`
4. **crosscheck** — ask each worker individually what's missing → re-transfer any gaps

---

## Auto-start

| Node | Mechanism | Effect |
|---|---|---|
| Server | `/Library/LaunchDaemons/` plist + `launchctl bootstrap system` | Runs `smoltorrent_startup.sh` at boot → waits for Tailscale → `launch.sh` |
| Each Pi | `systemd` template unit `smoltorrent-worker@{rank}.service` | Starts `worker.py` at boot, restarts on crash |

---

## Observability

Prometheus + Grafana + Loki run in Docker on the Server. Three metric sources feed the dashboard:

| Source | Endpoint | What it exposes |
|---|---|---|
| Master API | `localhost:8000/metrics/` | Transfer bytes, duration histograms, op counts, errors by rank |
| Pi workers | `<pi-ip>:920{rank}/metrics` | Per-worker store/send bytes, duration, op counts, errors |
| node_exporter | `<node>:9100/metrics` | CPU, memory, disk, load, temperature (all 5 nodes) |
| boot_exporter | `<node>:9101/metrics` | `smoltorrent_boot_time_ms` — OS boot timestamp (macOS + Linux) |

node_exporter and boot_exporter run on all 5 nodes. On the Server, both are registered as LaunchDaemons by `bash scripts/launch.sh --daemons`. On Pis, both are deployed as systemd units by `launch.sh` via SSH. See [monitoring/README.md](monitoring/README.md) for setup and the full metrics reference.

---

## Why each file lives where it does

| File | Rationale |
|---|---|
| `algorithms/SyncPS/worker.py` | Deployed to Pis — must not import MLX. Standalone TCP server + mDNS advertiser. |
| `backend/api.py` | Master only. HTTP so CLI and watcher share one entry point without duplicating TCP logic. |
| `watcher/watch.py` | Master only. Separated from API so it can be restarted independently. |
| `networking/send_receive.py` | Shared by both master and workers — one place for TCP framing. |
| `networking/ssh_manager.py` | Master only. Manages the fenced block in `~/.ssh/config` written by `discover`. |
| `discovery/grove/_mdns.py` | zeroconf logic isolated in grove — keeps discovery transport details out of the public API. |
| `discovery/grove/tui.py` | Textual TUI copied from smolcluster grove; `WorkerPickerApp` appended for smoltorrent node picking. |
| `discovery/grove/transport/p2p.py` | AirDrop/AWDL via Swift helper — macOS-only, loaded lazily. |
| `utils/common_utils.py` | Shared tensor ops, platform-branched on `_IS_MAC`. |
| `utils/shard_ops.py` | Separated so watcher, `main.py`, and tests all use the same HTTP path without duplicating httpx boilerplate. |
