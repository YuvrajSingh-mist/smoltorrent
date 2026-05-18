# smoltorrent

Distributed ML checkpoint sharding across a Raspberry Pi cluster, coordinated from a macOS master. Shards `.safetensors` checkpoints across workers over TCP with SHA-256 verification, replication factor 2, and automatic watcher sync.

**[→ Full documentation & setup guide](https://yuvrajsingh-mist.github.io/smoltorrent/)**

```
Master (Mac mini / Apple Silicon)
  ├── FastAPI server   backend/api.py          ← /store-shard, /gather-shards, /discover
  ├── Watcher daemon   watcher/watch.py         ← auto-syncs new checkpoints
  ├── Discovery        discovery/               ← mDNS + AirDrop device discovery
  └── Workers × N      algorithms/SyncPS/worker.py  ← TCP listener on each Pi
```

---

## Quick setup

### 1. Server (macOS)

```bash
brew install yq uv
git clone https://github.com/YuvrajSingh-mist/smoltorrent
cd smoltorrent && uv sync
```

### 2. Each Pi

Follow the [cluster setup guide](https://www.smolhub.com/posts/raspberry-pi-cluster-setup-guide) to get SSH access, then:

```bash
sudo apt update && sudo apt install -y python3.13 python3.13-venv curl git
```

> `uv`, `tmux`, and `node_exporter` are installed automatically on Pis by `launch.sh`.

### 3. Configure

Edit `configs/config.yaml` — set `ckpt_root` and each worker's `host`, `ip`, `port`, `rank`.
The `host` value must match your `~/.ssh/config` alias exactly.

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

### 4. Launch

```bash
bash scripts/launch.sh
```

Rsyncs code to all Pis, installs deps, starts API + watcher + workers in tmux.

---

## Usage

```bash
# Store a checkpoint across Pi workers
python main.py store --ckpt-path ~/smolcluster/checkpoints/Qwen2.5-0.5B/run1/latest/model.safetensors

# Reassemble from shards
python main.py gather --ckpt-path ~/smolcluster/checkpoints/Qwen2.5-0.5B/run1/latest/model.safetensors

# Find workers on the network (mDNS)
curl http://<master-ip>:8000/discover
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
