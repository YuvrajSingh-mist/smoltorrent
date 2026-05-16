# smoltorrent

Distributed ML checkpoint sharding across a Raspberry Pi cluster, coordinated from a macOS master. Shards `.safetensors` checkpoints across workers over Tailscale, reassembles them on demand, and auto-syncs new checkpoints as they appear.

```
Master (Mac Mini)
  ├── FastAPI server   backend/api.py          ← /store-shard, /gather-shards
  ├── Watcher daemon   watcher/watch.py         ← auto-syncs new checkpoints
  └── Workers × 4      algorithms/SyncPS/worker.py  ← TCP listener on each Pi
```

---

## How it works

**Store** — the API loads a checkpoint, splits tensors evenly into N shards (one per worker), computes a SHA-256 checksum per shard, and sends each over TCP. Workers verify the checksum and write the shard to disk with a `.checksum` sidecar. Failed sends retry with exponential backoff.

**Gather** — the API pulls each shard from its worker, saves locally, merges all shards back into one `.safetensors` file, and writes `merged.safetensors` next to the original checkpoint.

**Watcher** — monitors `ckpt_root` for new `.safetensors` files. On each trigger:
1. `file_sync` — asks all workers what they have (intersection)
2. `checksum_sync` — validates SHA-256 on every shared file
3. `transfer` — pushes any missing or corrupted files
4. `crosscheck` — after transfer, confirms every worker has every shard; re-transfers anything still missing

**Wire format** — safetensors only. MLX arrays (Mac) are converted to torch tensors before sending; workers store and return torch tensors; the master converts back to MLX on receive.

---

## Cluster topology

| Node | Host | IP | Port | Rank |
|---|---|---|---|---|
| Mac Mini (master) | localhost | 100.78.120.114 | 8000 (API) | 0 |
| pi4-1 | pi4-1 | 100.68.124.90 | 5001 | 1 |
| pi4-2 | pi4-2 | 100.79.150.107 | 5002 | 2 |
| pi4-3 | pi4-3 | 100.105.164.35 | 5003 | 3 |
| pi4-4 | pi4-4 | 100.77.162.23 | 8004 | 4 |

Network: Tailscale VPN over 100 Mbps Ethernet. ~2 min per 942 MB checkpoint per worker (parallel across all 4).

---

## Requirements

| Dependency | Where |
|---|---|
| Python ≥ 3.13 | All nodes |
| [uv](https://github.com/astral-sh/uv) | All nodes (auto-installed by launcher) |
| tmux ≥ 3.0 | All nodes (auto-installed by launcher) |
| [yq](https://github.com/mikefarah/yq) | Master only |
| SSH key-based auth | Master → all workers (`~/.ssh/smolcluster_key`) |
| Tailscale | All nodes |

Platform split:
- **Master (macOS)** — MLX for tensor ops; converts to torch before sending over wire
- **Workers (Raspberry Pi / Linux)** — `torch` + `safetensors.torch`; MLX never imported

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/YuvrajSingh-mist/smoltorrent
cd smoltorrent
uv sync
```

### 2. Configure

Edit `configs/config.yaml` — set `ckpt_root` and worker IPs/ports to match your cluster.

### 3. SSH access

```bash
ssh-keygen -t ed25519 -f ~/.ssh/smolcluster_key
ssh-copy-id -i ~/.ssh/smolcluster_key.pub lab-pi4-1@pi4-1
# repeat for each worker
```

### 4. Launch

```bash
bash scripts/launch.sh
```

This rsyncs the codebase to every Pi, installs dependencies, kills stale tmux sessions, then starts the API, watcher, and all 4 workers.

---

## Usage

### Store a checkpoint

```bash
python main.py store --ckpt-path ~/smolcluster/checkpoints/Qwen2.5-0.5B/run1/latest/model.safetensors
```

### Gather (reassemble) a checkpoint

```bash
python main.py gather --ckpt-path ~/smolcluster/checkpoints/Qwen2.5-0.5B/run1/latest/model.safetensors
# merged.safetensors is written to the same directory
```

### Watch logs

```bash
# API
tmux attach -t syncps_api

# Watcher
tmux attach -t syncps_watcher

# Worker (SSH first)
ssh pi4-1
tmux attach -t syncps_worker_1

# All logs
tail -f logging/cluster-logs/*.log
```

---

## launch.sh flags

| Flag | What it does |
|---|---|
| *(none)* | Full launch: rsync → deps → API + watcher + all workers |
| `--dry-run` | Print what would happen, no SSH or launches |
| `--api-only` | Heartbeat-check workers, then start API only (skip watcher + workers) |
| `--workers 1,3` | Launch only the specified worker ranks (leave others untouched) |
| `--ext .safetensors,.pth` | Override file extensions the watcher monitors |
| `--daemons` | Register auto-start at login (see below) |

---

## Auto-start on macOS

macOS has three ways to run background scripts automatically:

| Mechanism | Where | Runs as | When |
|---|---|---|---|
| **LaunchDaemon** | `/Library/LaunchDaemons/` | root | at boot, before login |
| **LaunchAgent** | `~/Library/LaunchAgents/` | user | on login |
| **Login Item (.app)** | System Settings | user | on login |

smoltorrent uses a **Login Item** (the only reliable path on macOS 26 Tahoe — LaunchAgents are silently ignored and LaunchDaemon `launchctl` APIs are broken in the beta):

```
login → ~/Applications/SmolTorrent Startup.app
            → scripts/smoltorrent_startup.sh
                → pings pi4-1 until Tailscale is up (5 min timeout)
                → bash scripts/launch.sh
                    → rsync + deps + tmux sessions on all nodes
```

### Register auto-start

```bash
bash scripts/launch.sh --daemons
```

Then follow the one manual step it prints:

> System Settings → General → Login Items & Extensions → + → `~/Applications/SmolTorrent Startup.app`

Logs after reboot:
```bash
tail -f /tmp/smoltorrent-startup.log
```

---

## Shard layout

On each Pi worker:
```
~/Desktop/smoltorrent/shards/worker_{rank}/
  {model}/{experiment}/{run}/latest/
    shard.safetensors
    shard.checksum        ← SHA-256 sidecar for corruption detection
```

On the master (local cache after gather):
```
~/smoltorrent/shards/worker_{rank}/
  {model}/{experiment}/{run}/latest/
    shard.safetensors
```

Merged output (written by gather):
```
~/smolcluster/checkpoints/{model}/{experiment}/{run}/latest/
  model.safetensors      ← original training checkpoint
  merged.safetensors     ← reassembled from Pi shards
```

---

## Project layout

```
smoltorrent/
├── algorithms/SyncPS/
│   └── worker.py            # TCP listener: store_shard, send_shard, sync,
│                            #   checksum_sync, all_shards_present
├── backend/
│   └── api.py               # FastAPI: /store-shard, /gather-shards
├── watcher/
│   └── watch.py             # Watchdog daemon: file_sync → checksum_sync
│                            #   → transfer → crosscheck
├── networking/
│   └── send_receive.py      # Length-prefixed TCP framing + bandwidth metrics
├── utils/
│   ├── common_utils.py      # shard_to_bytes/from_bytes, chunk_data, merge_shards
│   ├── shard_ops.py         # HTTP client wrappers for store/gather API calls
│   ├── check_workers.py     # TCP heartbeat check against all workers
│   ├── log_utils.py         # Per-component cluster logging
│   └── network_metrics.py   # Send/recv bandwidth and latency tracking
├── scripts/
│   ├── launch.sh            # Cluster orchestrator (rsync → deps → tmux)
│   └── smoltorrent_startup.sh  # Boot wrapper: wait for Tailscale → launch.sh
├── test/
│   └── test_received_model_inference.py  # Integration: load gathered weights, run MLX inference
├── configs/
│   └── config.yaml          # Cluster topology + ckpt_root
├── main.py                  # CLI: store / gather
└── pyproject.toml
```

---

## Configuration reference

| Key | Description |
|---|---|
| `ckpt_root` | Root directory the watcher monitors and gather saves into |
| `num_workers` | Number of Pi workers |
| `devices_config.master` | Master host, IP, rank (always 0), port |
| `devices_config.workers` | Per-worker: host, IP, rank (1…N), port |
| `n_chunks` | Shards per checkpoint (should equal `num_workers`) |

---

## License

See [LICENSE](LICENSE).
