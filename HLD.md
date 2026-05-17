# SmolTorrent — High-Level Design

---

## What the system does

SmolTorrent distributes ML checkpoint files across a cluster of Raspberry Pi workers and reassembles them on demand. The Mac Mini is the master — it holds the original checkpoints, runs the API and watcher, and coordinates all operations. The Pis are dumb storage workers — they receive shards, hold them on disk, and serve them back on request.

```
  Training script / user
         │
         │  writes model.safetensors
         ▼
  ~/smolcluster/checkpoints/   (ckpt_root on Mac)
         │
         │  watcher detects new file
         ▼
  FastAPI  (backend/api.py, port 8000)
    │   │   │   │
    ▼   ▼   ▼   ▼   TCP (Tailscale VPN, ~100 Mbps)
   Pi1 Pi2 Pi3 Pi4   (algorithms/SyncPS/worker.py)
  rank1 rank2 rank3 rank4
```

---

## Why this topology

- **Master orchestrates, workers store** — the Mac has MLX for tensor ops; Pis run pure torch. Keeping orchestration on the Mac avoids cross-framework complexity on workers.
- **Tailscale VPN** — Pis are behind NAT; Tailscale gives every node a stable routable IP without port-forwarding or a public server.
- **Safetensors as wire format** — the only format that carries shape + dtype + tensor name across both MLX (Mac) and torch (Pi) without importing the other framework. Pickle embeds the originating class; numpy fails on bfloat16.

---

## Three user-facing operations

| Operation | Trigger | What happens |
|---|---|---|
| **Store** | `python main.py store --ckpt-path <path>` or watcher auto-detect | Split checkpoint → push shards to all 4 Pis |
| **Gather** | `python main.py gather --ckpt-path <path>` | Pull shards from all 4 Pis → merge → write `merged.safetensors` |
| **Watch** | `watcher/watch.py` daemon (always running) | Auto-detect new checkpoints, store them, crosscheck all workers |

---

## Components and their roles

| Component | Where it runs | Role |
|---|---|---|
| `backend/api.py` | Mac, port 8000 | HTTP API — orchestrates store and gather |
| `watcher/watch.py` | Mac, daemon | Watches `ckpt_root`, triggers store automatically |
| `algorithms/SyncPS/worker.py` | Each Pi | TCP listener — stores and serves shards |
| `networking/send_receive.py` | Both | Zero-copy TCP framing shared by master and workers |
| `utils/common_utils.py` | Both | Tensor ops: chunk, serialize, deserialize, merge |
| `utils/shard_ops.py` | Mac | HTTP client wrappers that call the API |
| `scripts/launch.sh` | Mac (run manually) | rsync → deps → start all processes in tmux |
| `scripts/install_worker_service.sh` | Mac (run once) | Install systemd auto-restart service on each Pi |

---

## Store flow (high level)

```
Training writes checkpoint
  → watcher detects it
  → watcher calls POST /store-shard
  → API loads tensors, splits into 4 chunks
  → sends one chunk to each Pi in parallel
  → each Pi writes shard.safetensors + shard.checksum to disk
  → API streams progress back → watcher crosschecks all workers
```

## Gather flow (high level)

```
User runs: python main.py gather --ckpt-path <path>
  → calls POST /gather-shards
  → API sends ("send_shard", rank, rel_path) to each Pi in parallel
  → each Pi reads its shard from disk, sends bytes back
  → API merges all 4 chunks → writes merged.safetensors
```

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
| Mac | `/Library/LaunchDaemons/` plist + `launchctl bootstrap system` | Runs `smoltorrent_startup.sh` at boot → waits for Tailscale → `launch.sh` |
| Each Pi | `systemd` template unit `smoltorrent-worker@{rank}.service` | Starts `worker.py` at boot, restarts on crash |

---

## Why each file lives where it does

| File | Rationale |
|---|---|
| `algorithms/SyncPS/worker.py` | Deployed to Pis — must not import MLX. Standalone TCP server. |
| `backend/api.py` | Master only. HTTP so CLI and watcher share one entry point without duplicating TCP logic. |
| `watcher/watch.py` | Master only. Separated from API so it can be restarted independently. |
| `networking/send_receive.py` | Shared by both master and workers — one place for TCP framing. |
| `utils/common_utils.py` | Shared tensor ops, platform-branched on `_IS_MAC`. |
| `utils/shard_ops.py` | Separated so watcher, `main.py`, and tests all use the same HTTP path without duplicating httpx boilerplate. |
