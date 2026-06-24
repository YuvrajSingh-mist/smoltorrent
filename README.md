# smoltorrent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Distributed ML checkpoint sharding across a Raspberry Pi cluster, coordinated from a macOS master. Shards `.safetensors` checkpoints across workers over TCP with SHA-256 verification, replication factor 2, automatic watcher sync, and zero-config device discovery over mDNS and AirDrop.

*This is an educational project built to learn distributed systems concepts hands-on.*

| | |
|---|---|
| Full documentation & setup guide | [smoltorrent docs](https://yuvrajsingh-mist.github.io/smoltorrent/) |
| How it was built | [smolhub.com/posts/smoltorrent](https://www.smolhub.com/posts/smoltorrent/) |
| Beginner's guide to mDNS, Zeroconf & AWDL | [smolhub.com/posts/beginners-guide-to-mdns](https://www.smolhub.com/posts/beginners-guide-to-mdns/) |

```
Master (Mac mini / Apple Silicon)
  ├── FastAPI server   backend/api.py          ← /store-shard, /gather-shards, /discover
  ├── Watcher daemon   watcher/watch.py         ← auto-syncs new checkpoints
  ├── Discovery        discovery/               ← mDNS + AirDrop device discovery
  ├── SSH manager      networking/ssh_manager.py← writes ~/.ssh/config managed block
  └── Workers × N      algorithms/SyncPS/worker.py  ← TCP listener + mDNS advertiser on each Pi
```

## Setup

Full setup guide (standalone bootstrap, SSH cluster mode, auto-start, monitoring) → **[yuvrajsingh-mist.github.io/smoltorrent/setup.html](https://yuvrajsingh-mist.github.io/smoltorrent/setup.html)**

Once setup is done, bring the cluster up:

```bash
# On the master
grove start -n 4

# On each worker
grove join
```

Workers find the master via mDNS, register, and start automatically. Once all N workers have joined, the API + watcher start on the coordinator.

**If you have code changes to push to workers first** (contributor workflow):

`launch.sh` uses `configs/dev-config.yaml` - fill in your SSH aliases and Tailscale IPs there before running.

```bash
bash scripts/launch.sh        # rsync code to all workers (skips configs/)
bash scripts/grove_launch.sh  # restart API + watcher
```

## Usage

> **Recommended:** point `ckpt_root` in `config.yaml` to your checkpoint directory and let the watcher handle everything automatically. The watcher runs the full pipeline - `file_sync → checksum_sync → transfer → crosscheck` - which includes SHA-256 verification, retries, and a final crosscheck to confirm every worker received every shard. The `store` and `gather` commands below skip the crosscheck step, so they're best for one-off manual operations only.

```bash
# Step 1 - bring the cluster up:
grove start -n 4   # coordinator: advertise, wait for 4 workers
grove join         # (on each worker) TUI → select master → auto-registers
                   # once all N workers join, API + watcher start automatically

# (contributors) push code changes to workers first:
bash scripts/launch.sh

# ── cluster is now up, API server running at localhost:8000 ──────────────────

# Step 2 - store / gather (prefer grove CLI over curl):
grove store --ckpt-path ~/smolcluster/checkpoints/Qwen2.5-0.5B/run1/latest/model.safetensors
grove gather --ckpt-path ~/smolcluster/checkpoints/Qwen2.5-0.5B/run1/latest/model.safetensors

# Discover live workers (mDNS)
curl http://<master-ip>:8000/discover
```

> `grove store` and `grove gather` POST to the local API server at `localhost:8000`. The server must already be running (started by `grove start` or `launch.sh`) before either command works.

### Direct API (curl / Python)

`grove store` and `grove gather` are the recommended way to run store and gather. If you need to script against the API directly or integrate from Python, you can also hit the endpoints without the CLI:

**Store:**
```bash
curl -N -X POST \
  "http://localhost:8000/store-shard?ckpt_path=/abs/path/to/model.safetensors"
# Loaded 148 tensors (676.1 MB) - chunking into 4 shards
#   ✓ rank 1 (pi4-1) [round 0]
#   ✓ rank 2 (pi4-2) [round 0]
#   ...
# Done: 8/8 sends (2x replicated) → run1/latest
```

**Gather:**
```bash
curl -N -X POST \
  "http://localhost:8000/gather-shards?ckpt_path=/abs/path/to/model.safetensors"
#   ✓ shard 0 - saved → .../shards/worker_1/.../shard.safetensors
#   ✓ shard 1 - saved → .../shards/worker_2/.../shard.safetensors
#   ...
# Done: saved → /abs/path/to/merged.safetensors
```

`ckpt_path` must be absolute and under `ckpt_root` from `config.yaml`. Use `-N` with curl to stream output as it arrives. In Python use `httpx.Client(timeout=None)` with `client.stream()` + `iter_lines()`. Full API reference: **[docs →](https://yuvrajsingh-mist.github.io/smoltorrent/docs.html)**

## Redundancy

Every shard is stored on two workers. Store sends two rounds:
- **Round 0** - shard `i` → `workers[i]`
- **Round 1** - shard `i` → `workers[(i+1) % n]`

If a worker is unreachable during gather, the API automatically falls back to the worker that holds the round-1 replica. No data loss as long as no two adjacent workers fail simultaneously.

## Discoverability

Workers advertise themselves over mDNS (`_smoltorrent._tcp.local.`) on startup. The master runs a parallel mDNS + AirDrop/AWDL scan. No static IPs needed - workers are found by hostname and rank automatically.

```bash
# REST endpoint - returns all live workers with ip, port, rank, hostname
curl http://<master-ip>:8000/discover?timeout=10
```

## Optional

| Feature | Command |
|---|---|
| Pi auto-start on reboot | `bash scripts/install_worker_service.sh` |
| Server auto-start on reboot | `bash scripts/bootstrap.sh` (registers LaunchDaemons - run once) |
| Monitoring auto-start on reboot | `bash scripts/install_worker_service.sh --monitoring-daemon` |
| Monitoring (Prometheus + Grafana) | `cd monitoring && docker compose up -d` - see [Monitoring](#monitoring-prometheus--grafana--loki) section |

### Auto-start on reboot (macOS 26 Tahoe)

`bash scripts/bootstrap.sh` handles daemon registration as part of one-time setup. It does the following:

1. Copies `scripts/smoltorrent_startup.sh` to `/usr/local/bin/` (system daemons can't read `~/Desktop`, `~/Documents`, or `~/Downloads` - keep code and data outside those folders)
2. Writes a plist to `/Library/LaunchDaemons/com.smoltorrent.startup.plist`
3. Registers it with:

```bash
sudo launchctl enable system/com.smoltorrent.startup
sudo launchctl bootstrap system /Library/LaunchDaemons/com.smoltorrent.startup.plist
```

**Why not `~/Library/LaunchAgents/`?** macOS 26 silently ignores user-level LaunchAgents unless registered via `SMAppService` from a Swift app. Use `/Library/LaunchDaemons/` instead.

**If bootstrap fails (error 5):**

```bash
sudo launchctl bootout system/com.smoltorrent.startup 2>/dev/null || true
sudo launchctl enable system/com.smoltorrent.startup
sudo launchctl bootstrap system /Library/LaunchDaemons/com.smoltorrent.startup.plist
```

**[Full setup guide with all options →](https://yuvrajsingh-mist.github.io/smoltorrent/setup.html)**

## Monitoring (Prometheus + Grafana + Loki)

All monitoring runs in Docker on the master node - no SSH needed.

**First-time setup (regenerates `prometheus.yml` from config and starts all containers):**

```bash
bash scripts/launch_monitoring.sh
```

This is the canonical way to start monitoring - it regenerates `monitoring/prometheus/prometheus.yml` from `configs/config.yaml` (worker IPs, ports) before bringing containers up. Run it whenever the cluster topology changes.

**Start / stop individual containers:**

```bash
cd monitoring

# start everything
docker compose up -d

# stop everything
docker compose down

# restart a single service
docker compose restart prometheus

# check container status
docker compose ps

# tail logs for a service
docker compose logs -f grafana
```

**URLs:**

> **Accessing from a different machine (e.g. your MacBook when monitoring runs on mini1 / a remote node):**
> Replace `localhost` with the IP of the machine running Docker. For example, if monitoring runs on mini1 at `10.10.0.1`, use `http://10.10.0.1:3000` for Grafana.
> The ports below are bound to `0.0.0.0` in `docker-compose.yml`, so any machine on the same network or VPN can reach them directly.

| Service | URL (local) | Remote (example) | Login |
|---|---|---|---|
| Grafana | http://localhost:3000 | http://\<master-ip\>:3000 | `admin` / `smoltorrent` |
| Prometheus | http://localhost:9091 | http://\<master-ip\>:9091 | - |
| Loki | http://localhost:3100 | http://\<master-ip\>:3100 | - |

**Pi workers - ship logs to Loki:**

Once the cluster is up (`grove start` / `grove join`), `configs/config.yaml` has all the IPs and ranks. One command installs and starts Promtail on every Pi automatically:

```bash
bash scripts/launch_monitoring.sh --install-pi-promtail
```

It reads `configs/config.yaml`, generates a per-Pi config (filling in Loki IP, rank, SSH alias, and Linux username via `whoami` on each Pi), installs Promtail as a systemd service, and starts it. Promtail survives reboots automatically.

To target specific workers only:

```bash
bash scripts/launch_monitoring.sh --install-pi-promtail --workers 1,3
```

Pi logs appear in Grafana → Explore → Loki:

```
{job="smoltorrent"}              # all nodes
{job="smoltorrent", node="pi4-2"} # one Pi
{job="smoltorrent", level="ERROR"} # errors only
```

## License

See [LICENSE](LICENSE).
