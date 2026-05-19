# smoltorrent

Distributed ML checkpoint sharding across a Raspberry Pi cluster, coordinated from a macOS master. Shards `.safetensors` checkpoints across workers over TCP with SHA-256 verification, replication factor 2, automatic watcher sync, and zero-config device discovery over mDNS and AirDrop.

**[→ Full documentation & setup guide](https://yuvrajsingh-mist.github.io/smoltorrent/)**

```
Master (Mac mini / Apple Silicon)
  ├── FastAPI server   backend/api.py          ← /store-shard, /gather-shards, /discover
  ├── Watcher daemon   watcher/watch.py         ← auto-syncs new checkpoints
  ├── Discovery        discovery/               ← mDNS + AirDrop device discovery
  ├── SSH manager      networking/ssh_manager.py← writes ~/.ssh/config managed block
  └── Workers × N      algorithms/SyncPS/worker.py  ← TCP listener + mDNS advertiser on each Pi
```

---

## Quick setup

### 1. Server (macOS)

```bash
brew install yq uv
git clone https://github.com/YuvrajSingh-mist/smoltorrent
cd smoltorrent && uv sync
```

Add the `grove` command to your shell (one time):

```bash
echo 'export PATH="$HOME/smoltorrent/.venv/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

---

### Option A — Quick start / testing (no SSH setup needed)

Workers discover the master over mDNS and self-register. No `config.yaml` editing, no SSH config required.

**Each worker** (Pi or Mac mini) — clone and install once:

```bash
git clone https://github.com/YuvrajSingh-mist/smoltorrent
cd smoltorrent && uv pip install -e .
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

---

### Option B — Production / serious runs (SSH-based)

Full cluster management via `launch.sh` — rsyncs code to all Pis, installs deps, starts everything in tmux.

**Prerequisites:** SSH key access to each Pi. Add aliases to `~/.ssh/config`:

```
Host pi4-1
    HostName <pi-ip>
    User <pi-user>
    IdentityFile ~/.ssh/<your-key>
    IdentitiesOnly yes
```

**Edit `configs/config.yaml`** — set `ckpt_root` and one entry per worker (the `host` must match your SSH alias exactly):

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

**Launch:**

```bash
bash scripts/launch.sh
```

Rsyncs code to all Pis, installs deps, starts API + watcher + workers in tmux.

> **Warning:** both `launch.sh` and `grove start` forcibly free ports before starting. On the coordinator: ports **8000** (API) and **8001** (watcher metrics). On each worker Pi: port **9200+rank** (Prometheus metrics, e.g. 9201–9204). Any process already using those ports will be killed.

---

## Usage

> **Recommended:** point `ckpt_root` in `config.yaml` to your checkpoint directory and let the watcher handle everything automatically. The watcher runs the full pipeline — `file_sync → checksum_sync → transfer → crosscheck` — which includes SHA-256 verification, retries, and a final crosscheck to confirm every worker received every shard. The `store` and `gather` commands below skip the crosscheck step, so they're best for one-off manual operations only.

```bash
# Step 1 — bring the cluster up (one of two options):

# Option A — mDNS auto-discovery:
grove start -n 4   # master: advertise, wait for 4 workers
grove join         # (on each worker Pi) TUI → select master → auto-registers
                   # once all N workers join, API + watcher start automatically

# Option B — SSH-based (edit configs/config.yaml first):
bash scripts/launch.sh

# ── cluster is now up, API server running at localhost:8000 ──────────────────

# Step 2 — store / gather (prefer grove CLI over curl):
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
# Loaded 148 tensors (676.1 MB) — chunking into 4 shards
#   ✓ rank 1 (pi4-1) [round 0]
#   ✓ rank 2 (pi4-2) [round 0]
#   ...
# Done: 8/8 sends (2x replicated) → run1/latest
```

**Gather:**
```bash
curl -N -X POST \
  "http://localhost:8000/gather-shards?ckpt_path=/abs/path/to/model.safetensors"
#   ✓ shard 0 — saved → .../shards/worker_1/.../shard.safetensors
#   ✓ shard 1 — saved → .../shards/worker_2/.../shard.safetensors
#   ...
# Done: saved → /abs/path/to/merged.safetensors
```

`ckpt_path` must be absolute and under `ckpt_root` from `config.yaml`. Use `-N` with curl to stream output as it arrives. In Python use `httpx.Client(timeout=None)` with `client.stream()` + `iter_lines()`. Full API reference: **[docs →](https://yuvrajsingh-mist.github.io/smoltorrent/docs.html)**

---

## Redundancy

Every shard is stored on two workers. Store sends two rounds:
- **Round 0** — shard `i` → `workers[i]`
- **Round 1** — shard `i` → `workers[(i+1) % n]`

If a worker is unreachable during gather, the API automatically falls back to the worker that holds the round-1 replica. No data loss as long as no two adjacent workers fail simultaneously.

---

## Discoverability

Workers advertise themselves over mDNS (`_smoltorrent._tcp.local.`) on startup. The master runs a parallel mDNS + AirDrop/AWDL scan. No static IPs needed — workers are found by hostname and rank automatically.

```bash
# REST endpoint — returns all live workers with ip, port, rank, hostname
curl http://<master-ip>:8000/discover?timeout=10
```

---

## Optional

| Feature | Command |
|---|---|
| Pi auto-start on reboot | `bash scripts/install_worker_service.sh` |
| Server auto-start on reboot | `bash scripts/launch.sh --daemons` |
| Monitoring (Prometheus + Grafana) | `cd monitoring && docker compose up -d` — no SSH needed |

**[Full setup guide with all options →](https://yuvrajsingh-mist.github.io/smoltorrent/setup.html)**

---

## License

See [LICENSE](LICENSE).
