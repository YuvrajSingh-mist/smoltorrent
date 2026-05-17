# smoltorrent

Distributed ML checkpoint sharding across a Raspberry Pi cluster, coordinated from a macOS master. Shards `.safetensors` checkpoints across workers over Tailscale, reassembles them on demand, and auto-syncs new checkpoints as they appear.

```
Master (Server)
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
2. `checksum_sync` *(startup only)* — validates SHA-256 on every shared file; skipped on file-event triggers
3. `transfer` — pushes any missing or corrupted files
4. `crosscheck` — after transfer, confirms every worker has every shard; re-transfers anything still missing

Files detected while still being written go to a **pending list**. A dedicated polling thread (`_run_pending_loop`) re-evaluates them every 10 s and re-triggers the transfer loop once they become stable.

**Wire format** — safetensors only. MLX arrays (Server) are converted to torch tensors before sending; workers store and return torch tensors; the master converts back to MLX on receive.

---

## Cluster topology

| Node | Host | IP | Port | Rank |
|---|---|---|---|---|
| Server (master) | localhost | 100.78.120.114 | 8000 (API) | 0 |
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

### macOS 26 Tahoe — what's broken and what works

Tahoe broke the traditional auto-start APIs:

| What you try | What happens |
|---|---|
| `launchctl load com.smoltorrent.startup.plist` | SIGABRT exit 134 — API removed |
| `launchctl bootstrap gui/<UID> ...plist` | error 125 — GUI domain broken in beta |
| `~/Library/LaunchAgents/` plist | Silently ignored — needs SMAppService from Swift |
| **`/Library/LaunchDaemons/` + `sudo launchctl enable` + `sudo launchctl bootstrap system`** | **Works** |

**TCC caveat**: macOS blocks system daemons (running as root) from accessing `~/Desktop`, `~/Documents`, and `~/Downloads`. Keep code and checkpoint data outside those directories — that's why everything lives under `~/smoltorrent/` and `~/smolcluster/`.

### Boot flow

```
boot
 └── /Library/LaunchDaemons/com.smoltorrent.startup.plist   (registered by --daemons)
       └── /usr/local/bin/smoltorrent_startup.sh
             → pings pi4-1 every 5s until Tailscale is up (5 min timeout)
             → bash ~/smoltorrent/scripts/launch.sh
                   → rsync code to all Pis
                   → uv sync on all nodes
                   → tmux: API + watcher on master, worker_{rank} on each Pi
```

### Register auto-start

```bash
bash scripts/launch.sh --daemons
```

This copies `smoltorrent_startup.sh` to `/usr/local/bin/` (outside TCC-blocked paths), writes the plist below to `/Library/LaunchDaemons/`, then registers it:

```bash
sudo launchctl bootout system/com.smoltorrent.startup 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/com.smoltorrent.startup.plist
sudo launchctl enable system/com.smoltorrent.startup
```

The plist it writes:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.smoltorrent.startup</string>
    <key>UserName</key>
    <string>yuvrajsingh1</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/usr/local/bin/smoltorrent_startup.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>/tmp/smoltorrent-startup.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/smoltorrent-startup.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HOME</key>
        <string>/Users/yuvrajsingh1</string>
    </dict>
</dict>
</plist>
```

`UserName` makes it run as your user not root. `EnvironmentVariables` ensures Homebrew and uv are on PATH — without it the startup script can't find them.

### Verify it's registered

```bash
sudo launchctl print system/com.smoltorrent.startup
```

### Remove / uninstall

```bash
sudo launchctl bootout system/com.smoltorrent.startup 2>/dev/null || true
sudo rm -f /Library/LaunchDaemons/com.smoltorrent.startup.plist
sudo rm -f /usr/local/bin/smoltorrent_startup.sh

# Verify it's gone
sudo launchctl print system/com.smoltorrent.startup 2>&1
```

### If bootstrap fails with "Bootstrap failed: 5"

```bash
sudo launchctl bootout system/com.smoltorrent.startup 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/com.smoltorrent.startup.plist
sudo launchctl enable system/com.smoltorrent.startup
```

### Check if it ran successfully

```bash
cat /tmp/smoltorrent-startup.log
```

A successful run looks like:

```
[Sat May 16 05:37:31 IST 2026] smoltorrent_startup: waiting for Tailscale (100.68.124.90)...
[Sat May 16 05:37:31 IST 2026] smoltorrent_startup: network ready — launching cluster...
Project: /Users/yuvrajsingh1/smoltorrent
...
Launched syncps_api on local host localhost
Launched syncps_watcher on local host localhost
Launched syncps_worker_1 on remote host pi4-1
...
Launch complete.
```

Note: `launchctl print` may show `last exit code = 1` and `state = not running` — this is expected. The daemon runs once at boot, launches everything into tmux sessions, then exits. The cluster keeps running in tmux independently.

### Logs after reboot

```bash
tail -f /tmp/smoltorrent-startup.log
```

---

## Monitoring (Prometheus + Grafana + Loki)

All logs and metrics in one place — Server API, watcher, and all 4 Pi workers.

```bash
cd monitoring
docker compose up -d
# Grafana → http://localhost:3000  (admin / smoltorrent)
```

Four dashboard sections — smoltorrent transfer metrics, Server system stats, Pi worker system stats + per-Pi smoltorrent metrics, and API server stats — plus a unified log stream from all nodes.

- **Master metrics** — `http://localhost:8000/metrics` (FastAPI + `prometheus_client`)
- **Pi worker metrics** — `http://<pi>:920{rank}/metrics` (per-worker `prometheus_client` server in `worker.py`)
- **System metrics** — `http://<node>:9100/metrics` (node_exporter on all 5 nodes)
- **Logs** — Promtail → Loki on Server (Docker) and all Pis (systemd)

See [monitoring/README.md](monitoring/README.md) for full setup, metrics reference, and dashboard panel guide.

---

## Pi worker auto-start (systemd)

The master auto-starts via LaunchDaemon (above), which SSHes into each Pi to launch `worker.py`. But if a Pi reboots independently later, its worker process is dead until you re-run `launch.sh`. To fix that, install a systemd service on every Pi:

```bash
# Install on all 4 workers (run from master)
bash scripts/install_worker_service.sh

# Specific ranks only
bash scripts/install_worker_service.sh --workers 1,3

# Custom SSH key
bash scripts/install_worker_service.sh --ssh-key ~/.ssh/my_key
```

This installs `/etc/systemd/system/smoltorrent-worker@.service` on each Pi, enables `smoltorrent-worker@<rank>`, and starts it immediately. On reboot the Pi brings up its worker automatically without waiting for the master.

**Useful commands (run from master):**

```bash
# Status
ssh -i ~/.ssh/smolcluster_key pi4-1 'systemctl status smoltorrent-worker@1'

# Live logs
ssh -i ~/.ssh/smolcluster_key pi4-1 'journalctl -u smoltorrent-worker@1 -f'

# Restart a worker manually
ssh -i ~/.ssh/smolcluster_key pi4-2 'sudo systemctl restart smoltorrent-worker@2'

# Uninstall from all workers
bash scripts/install_worker_service.sh --uninstall
```

> **Note:** `launch.sh` still kills and re-launches workers via tmux when you run it. The systemd service and the tmux session are independent — if tmux is running the worker, systemd's process will fail to bind the port and restart after 5 s. Run `install_worker_service.sh` when you want systemd as the primary keeper; use `launch.sh` alone when you want manual tmux control.

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
│                            #   + prometheus_client metrics on port 9200+rank
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
│   ├── launch.sh                  # Cluster orchestrator (rsync → deps → tmux)
│   ├── smoltorrent_startup.sh     # Boot wrapper: wait for Tailscale → launch.sh
│   └── install_worker_service.sh  # Install systemd service on Pi workers for auto-restart
├── test/
│   ├── test_shard_serialization.py       # Unit: shard_to_bytes, checksum, merge (no cluster needed)
│   ├── test_worker_commands.py           # Integration: all worker TCP commands
│   ├── test_watcher_logic.py             # Integration: watcher sync, crosscheck, file trigger
│   ├── test_pending_loop.py              # Integration: pending loop with real file sizes (~150–400 MB)
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
