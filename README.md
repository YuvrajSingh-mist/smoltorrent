# smoltorrent

Distributed ML checkpoint sharding across a Raspberry Pi cluster, coordinated from a macOS master. Shards `.safetensors` checkpoints across workers over TCP, reassembles them on demand, and auto-syncs new checkpoints as they appear.

```
Master (Server)
  ├── FastAPI server   backend/api.py          ← /store-shard, /gather-shards
  ├── Watcher daemon   watcher/watch.py         ← auto-syncs new checkpoints
  └── Workers × 4      algorithms/SyncPS/worker.py  ← TCP listener on each Pi
```

---

## Tested on

| Node | Hardware | OS | Python | RAM | Storage |
|---|---|---|---|---|---|
| Server (master) | Apple Mac mini M4 | macOS 26.2 Tahoe (arm64) | 3.13 | 16 GB | — |
| pi4-1 … pi4-4 | Raspberry Pi 4 Model B Rev 1.5 | Debian GNU/Linux 13 Trixie (aarch64, kernel 6.12) | 3.13.5 | 4 GB | 64 GB SD card |

Network: ~100 Mbps Ethernet between nodes (tested over Tailscale VPN).

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

All topology is defined in `configs/config.yaml` — no hostnames or IPs are hardcoded in the code.

| Node | Host | IP | Port | Rank |
|---|---|---|---|---|
| Server (master) | `localhost` | your server IP | 8000 (API) | 0 |
| Worker 1 | `<hostname>` | `<ip>` | 5001 | 1 |
| Worker 2 | `<hostname>` | `<ip>` | 5002 | 2 |
| … | … | … | … | … |
| Worker N | `<hostname>` | `<ip>` | 500N | N |

Network: nodes must be mutually reachable over TCP — local LAN, VPN, or any other network works. ~2 min per 942 MB checkpoint per worker (parallel across all N).

---

## Setup (from scratch)

### 1. Prerequisites — Server (macOS)

```bash
# Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Required tools
brew install yq          # YAML parser used by launch.sh
brew install uv          # Python package manager (also auto-installed on all nodes by launch.sh)


```

### 2. Prerequisites — each Pi

Follow the [Raspberry Pi cluster setup guide](https://www.smolhub.com/posts/raspberry-pi-cluster-setup-guide) to get your Pis networked and SSH-accessible, then run on each Pi:

```bash


# Python 3.13
sudo apt update && sudo apt install -y python3.13 python3.13-venv curl git
```

> `uv`, `tmux`, and `node_exporter` are installed automatically by `launch.sh` on first run — no need to install them manually on the Pis.


> **Important:** the `Host` alias in `~/.ssh/config` must exactly match the `host` field in `configs/config.yaml`. `launch.sh` uses those values directly as SSH targets — if they don't match, SSH will fail. If you followed the [cluster setup guide](https://www.smolhub.com/posts/raspberry-pi-cluster-setup-guide), use the same aliases the guide tells you to set up and mirror them in `config.yaml`.

Verify:

```bash
ssh pi4-1   # or whatever alias you chose
```

### 4. Clone and configure

```bash
git clone https://github.com/YuvrajSingh-mist/smoltorrent
cd smoltorrent
uv sync
```

Edit `configs/config.yaml` — set `ckpt_root` and each worker's `host`, `ip`, `port`, and `rank`. **The `host` value must match your `~/.ssh/config` alias exactly.**

```yaml
devices_config:
  master:
    - host: localhost
      ip: <your-server-ip>
      rank: 0
      port: 5000
  workers:
    - host: pi4-1          # must match Host alias in ~/.ssh/config
      ip: <ip>
      rank: 1
      port: 5001
    - host: pi4-2
      ip: <ip>
      rank: 2
      port: 5002
    # ... one entry per worker
```

### 5. Launch the cluster

```bash
bash scripts/launch.sh
```

This rsyncs the codebase to every Pi, installs `uv` + `tmux` + `node_exporter` on each node, then starts the API, watcher, and all 4 workers in tmux sessions.

### 6. Pi worker auto-start (optional but recommended)

So workers restart automatically if a Pi reboots independently:

```bash
bash scripts/install_worker_service.sh
```

### 7. Auto-start on server boot (optional but recommended)

So the entire cluster comes up after a server reboot — no manual intervention:

```bash
bash scripts/launch.sh --daemons
```

This registers two system LaunchDaemons that survive reboots:
- `com.smoltorrent.startup` — waits for network, then runs `launch.sh`
- `com.node-exporter` — keeps `node_exporter` running for Grafana system stats

### 8. Monitoring (optional)

Prometheus + Grafana + Loki in Docker:

```bash
# Install Docker via colima (macOS)
brew install colima docker docker-compose
colima start

# Copy and fill in credentials
cp monitoring/.env.example monitoring/.env
# edit monitoring/.env — set Gmail app password for alert emails

# Start the stack
cd monitoring && docker-compose up -d
# Grafana → http://localhost:3000  (admin / smoltorrent)
```

See [monitoring/README.md](monitoring/README.md) for full setup, alert configuration, and dashboard guide.

---

## Requirements summary

| Dependency | Where | How |
|---|---|---|
| Python ≥ 3.13 | All nodes | Manual on Pis; already on macOS |
| uv | All nodes | Auto-installed by `launch.sh` |
| tmux ≥ 3.0 | All nodes | Auto-installed by `launch.sh` |
| yq | Server only | `brew install yq` |
| node_exporter | All nodes | Auto-installed by `launch.sh` |
| Network (LAN/VPN) | All nodes | Nodes must reach each other over TCP |
| SSH key auth | Server → Pis | `ssh-copy-id` (see step 3) |
| Docker + colima | Server only | For monitoring only |

Platform split:
- **Server (macOS)** — MLX for tensor ops; converts to torch before sending over wire
- **Workers (Raspberry Pi OS / Linux)** — `torch` + `safetensors.torch`; MLX never imported

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
ssh <worker-hostname>
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

### Boot flow

```
boot
 ├── /Library/LaunchDaemons/com.node-exporter.plist         (registered by --daemons)
 │     └── node_exporter :9100  (always running — Grafana system stats)
 │
 └── /Library/LaunchDaemons/com.smoltorrent.startup.plist   (registered by --daemons)
       └── /usr/local/bin/smoltorrent_startup.sh
             → pings first worker every 5s until network is up (5 min timeout)
             → bash ~/smoltorrent/scripts/launch.sh
                   → rsync code to all Pis
                   → uv sync on all nodes
                   → tmux: API + watcher on master, worker_{rank} on each Pi
```

### Register auto-start

```bash
bash scripts/launch.sh --daemons
```

This does **two things** in one go:

1. Registers the smoltorrent startup LaunchDaemon (copies `smoltorrent_startup.sh` to `/usr/local/bin/`, writes plist to `/Library/LaunchDaemons/com.smoltorrent.startup.plist`)
2. Installs `node_exporter` via Homebrew (if not already installed) and registers it as a separate system LaunchDaemon (`/Library/LaunchDaemons/com.node-exporter.plist`) so Grafana system-stats panels (CPU, disk, memory, boot time) survive reboots

**`/Library/LaunchDaemons/com.smoltorrent.startup.plist`** — runs once at boot, launches the cluster:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.smoltorrent.startup</string>
    <key>UserName</key>
    <string>YOUR_USERNAME</string>  <!-- filled in automatically by launch.sh -->
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
        <string>/Users/YOUR_USERNAME</string>  <!-- filled in automatically by launch.sh -->
    </dict>
</dict>
</plist>
```

`UserName` runs it as your user not root. `EnvironmentVariables` puts Homebrew and uv on PATH — without it the script can't find them.

**`/Library/LaunchDaemons/com.node-exporter.plist`** — keeps node_exporter alive permanently for Grafana system stats:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.node-exporter</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/node_exporter</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/node-exporter.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/node-exporter.log</string>
</dict>
</plist>
```

`KeepAlive true` means launchd restarts it automatically if it crashes. Metrics available at `http://localhost:9100/metrics`.

### Verify both are registered

```bash
sudo launchctl print system/com.smoltorrent.startup
sudo launchctl print system/com.node-exporter

# Quick health check
curl -s http://localhost:9100/metrics | grep node_boot_time_seconds
```

### Remove / uninstall

```bash
# smoltorrent startup
sudo launchctl bootout system/com.smoltorrent.startup 2>/dev/null || true
sudo rm -f /Library/LaunchDaemons/com.smoltorrent.startup.plist
sudo rm -f /usr/local/bin/smoltorrent_startup.sh

# node_exporter
sudo launchctl bootout system/com.node-exporter 2>/dev/null || true
sudo rm -f /Library/LaunchDaemons/com.node-exporter.plist
```

### If bootstrap fails with "Bootstrap failed: 5"

```bash
# smoltorrent
sudo launchctl bootout system/com.smoltorrent.startup 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/com.smoltorrent.startup.plist
sudo launchctl enable system/com.smoltorrent.startup

# node_exporter
sudo launchctl bootout system/com.node-exporter 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/com.node-exporter.plist
sudo launchctl enable system/com.node-exporter
```

### Check if it ran successfully

```bash
cat /tmp/smoltorrent-startup.log
```

A successful run looks like:

```
[Sat May 16 05:37:31 IST 2026] smoltorrent_startup: waiting for network (<worker-ip>)...
[Sat May 16 05:37:31 IST 2026] smoltorrent_startup: network ready — launching cluster...
Project: /Users/<your-username>/smoltorrent
...
Launched syncps_api on local host localhost
Launched syncps_watcher on local host localhost
Launched syncps_worker_1 on remote host <worker-1-hostname>
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

## Pi worker systemd service

`install_worker_service.sh` installs `/etc/systemd/system/smoltorrent-worker@.service` on each Pi, enables `smoltorrent-worker@<rank>`, and starts it. On Pi reboot the worker comes up automatically without waiting for the server.

```bash
bash scripts/install_worker_service.sh            # all 4 workers
bash scripts/install_worker_service.sh --workers 1,3   # specific ranks
bash scripts/install_worker_service.sh --uninstall     # remove from all
```

**Useful commands:**

```bash
ssh <worker-1-hostname> 'systemctl status smoltorrent-worker@1'
ssh <worker-1-hostname> 'journalctl -u smoltorrent-worker@1 -f'
ssh <worker-2-hostname> 'sudo systemctl restart smoltorrent-worker@2'
```

> `launch.sh` kills and re-launches workers via tmux on every run regardless — systemd and tmux are independent. If both are running, systemd's process will fail to bind the port and retry after 5 s.

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
│   ├── smoltorrent_startup.sh     # Boot wrapper: wait for network → launch.sh
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
| `devices_config.workers` | Per-worker: host, IP, rank (1…N), port — **`host` must match the `Host` alias in `~/.ssh/config`** |
| `n_chunks` | Shards per checkpoint (should equal `num_workers`) |

---

## License

See [LICENSE](LICENSE).
