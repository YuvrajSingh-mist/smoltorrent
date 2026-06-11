# smoltorrent

Distributed ML checkpoint sharding across a Raspberry Pi cluster, coordinated from a macOS master. Shards `.safetensors` checkpoints across workers over TCP with SHA-256 verification, replication factor 2, automatic watcher sync, and zero-config device discovery over mDNS and AirDrop.

**[→ Full documentation & setup guide](https://yuvrajsingh-mist.github.io/smoltorrent/)** · **[→ How it was built](https://www.smolhub.com/posts/smoltorrent/)**

```
Master (Mac mini / Apple Silicon)
  ├── FastAPI server   backend/api.py          ← /store-shard, /gather-shards, /discover
  ├── Watcher daemon   watcher/watch.py         ← auto-syncs new checkpoints
  ├── Discovery        discovery/               ← mDNS + AirDrop device discovery
  ├── SSH manager      networking/ssh_manager.py← writes ~/.ssh/config managed block
  └── Workers × N      algorithms/SyncPS/worker.py  ← TCP listener + mDNS advertiser on each Pi
```

## Before you start: things to change for your cluster

Three files contain values specific to this setup that you must replace:

**1. `configs/config.yaml`** - the single source of truth for your cluster topology.
Edit every IP, hostname, and port to match your workers:
```yaml
devices_config:
  master:
  - host: localhost
    ip: <your-mac-tailscale-ip>   # run: tailscale ip -4
    rank: 0
    port: 5000
  workers:
  - host: <ssh-alias-for-worker-1>   # must match Host in ~/.ssh/config
    ip: <worker-1-tailscale-ip>
    rank: 1
    port: 5001
  # ... one entry per worker
```

**2. `scripts/smoltorrent_startup.sh`** - two lines at the top:
```bash
SMOLTORRENT_DIR="/Users/yuvrajsingh1/smoltorrent"   # ← change to where you cloned the repo
TAILSCALE_PROBE="100.68.124.90"                      # ← change to your first worker's Tailscale IP
```

**3. `monitoring/prometheus/prometheus.yml`** - every worker IP is hardcoded there.
Run `bash scripts/launch_monitoring.sh` instead of editing by hand - it regenerates the file from `configs/config.yaml`.

Everything else (SSH key path, usernames, ports) is read from `configs/config.yaml` at runtime.

## Setup

### 1. Clone on the server (macOS)

```bash
brew install yq uv
git clone https://github.com/YuvrajSingh-mist/smoltorrent
cd smoltorrent && uv sync
```

### 2. Configure `configs/config.yaml`

Set `ckpt_root` and add one entry per worker. The `host` value must match your SSH alias in `~/.ssh/config` exactly:

```yaml
ckpt_root: /path/to/checkpoints
devices_config:
  master:
    - host: localhost
      ip: <server-ip>
      rank: 0
      port: 5000
  workers:
    - host: pi4-1        # must match Host alias in ~/.ssh/config
      ip: <pi-ip>
      rank: 1
      port: 5001
    # ... one entry per worker
```

Add SSH aliases to `~/.ssh/config` (one per Pi):

```
Host pi4-1
    HostName <pi-ip>
    User <pi-user>
    IdentityFile ~/.ssh/<your-key>
    IdentitiesOnly yes
```

### 3. Bootstrap all nodes (run once)

Rsyncs code to every Pi and installs all dependencies (uv, tmux, node_exporter, venv, zeroconf, boot_exporter service):

```bash
bash scripts/bootstrap.sh
```

After this completes, every node has everything it needs. You can go straight to `grove` or `launch.sh` - no further setup needed.

### 4. Launch the cluster

```bash
bash scripts/launch.sh
```

Rsyncs latest code to workers (fast, no dep install), kills stale sessions, starts API + watcher + workers in tmux.

> **Warning:** `launch.sh` forcibly frees ports before starting. On the coordinator: ports **8000** (API) and **8001** (watcher metrics). On each worker Pi: port **9200+rank** (Prometheus metrics, e.g. 9201–9204). Any process already using those ports will be killed.

---

### No-SSH alternative: grove start / join

Once bootstrap has run on every node, you can skip `launch.sh` entirely and use grove instead. Good for testing and same-network setups - no SSH config, no manual `config.yaml` editing needed.

Add the `grove` command to your shell once:

```bash
# macOS (zsh)
echo 'export PATH="<path-to-smoltorrent>/.venv/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

# Linux / Pi (bash)
echo 'export PATH="<path-to-smoltorrent>/.venv/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

**Master:**
```bash
grove start -n 4
```

**Each worker:**
```bash
grove join
```

Workers find the master via mDNS TUI, register, and start automatically. The master writes `configs/config.yaml` and launches the API + watcher when all N workers have joined.

## Usage

> **Recommended:** point `ckpt_root` in `config.yaml` to your checkpoint directory and let the watcher handle everything automatically. The watcher runs the full pipeline - `file_sync → checksum_sync → transfer → crosscheck` - which includes SHA-256 verification, retries, and a final crosscheck to confirm every worker received every shard. The `store` and `gather` commands below skip the crosscheck step, so they're best for one-off manual operations only.

```bash
# Step 1 - bring the cluster up:

# SSH-based (primary - rsyncs latest code and starts everything; run bootstrap.sh first):
bash scripts/launch.sh

# No-SSH alternative (mDNS auto-discovery):
grove start -n 4   # master: advertise, wait for 4 workers
grove join         # (on each worker Pi) TUI → select master → auto-registers
                   # once all N workers join, API + watcher start automatically

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
| Server auto-start on reboot | `bash scripts/launch.sh --daemons` |
| Monitoring (Prometheus + Grafana) | `cd monitoring && docker compose up -d` - see [Monitoring](#monitoring-prometheus--grafana--loki) section |

### Auto-start on reboot (macOS 26 Tahoe)

`bash scripts/launch.sh --daemons` does the following:

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

**Start:**

```bash
cd monitoring
docker compose up -d
```

**URLs:**

| Service | URL | Login |
|---|---|---|
| Grafana | http://localhost:3000 | `admin` / `smoltorrent` |
| Prometheus | http://localhost:9090 | - |
| Loki | http://localhost:3100 | - |

**Useful commands:**

```bash
# check all containers are healthy
docker compose -f monitoring/docker-compose.yml ps

# tail logs for a service
docker compose -f monitoring/docker-compose.yml logs -f prometheus

# stop everything
docker compose -f monitoring/docker-compose.yml down
```

**Pi workers - ship logs to Loki:**

Once the cluster is up (`grove start` / `grove join` or `launch.sh`), `configs/config.yaml` has all the IPs and ranks. One command installs and starts Promtail on every Pi automatically:

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
