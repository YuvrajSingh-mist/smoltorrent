# smoltorrent

Distributed model sharding across heterogeneous edge devices using a synchronous parameter server (SyncPS). Built to run on Raspberry Pi clusters coordinated from a macOS master via a REST API.

```
Master (macOS)
  ├── REST API   (backend/api.py)         ← /store-shard, /gather-shards
  └── Workers × N (algorithms/SyncPS/worker.py  ← runs on each Pi)
```

---

## How it works

1. **Store** — `POST /store-shard`: The API on the master loads the model from `data_path`, splits it into `N` shards (one per worker), computes a SHA-256 checksum per shard, and sends each shard over TCP to its ranked Pi worker. Workers verify the checksum and save the shard to disk. Failed sends are retried with exponential backoff on a background thread.

2. **Workers** each run a TCP listener. On `store_shard` they verify the checksum, write the shard to `shards/incoming_shards/{model}/{worker-rank}/`, and ack back with the shard path. On `send_shard` they load the shard from disk and stream it back.

3. **Gather** — `POST /gather-shards`: The API connects to each worker, requests its shard (`send_shard`), receives the bytes, deserializes, merges all shards into one model, and writes it to `save_path`.

4. `main.py` is a CLI that runs three checks before triggering gather:
   - **Heartbeat** — TCP ping every worker
   - **Shard count** — SSH to each worker, count `.safetensors` files on disk
   - **Gather** — only proceeds if all shards are present

---

## Requirements

| Dependency | Where needed |
|---|---|
| Python ≥ 3.13 | All nodes |
| [uv](https://github.com/astral-sh/uv) | All nodes (auto-installed by launcher) |
| tmux ≥ 3.0 | All nodes (auto-installed by launcher) |
| [yq](https://github.com/mikefarah/yq) | Master only (config parsing) |
| SSH key-based auth | Master -> all workers |

Platform notes:
- Master (macOS): uses MLX for tensor operations; safetensors used as the cross-platform wire format
- Workers (Raspberry Pi / Linux): use `torch` + `safetensors.torch`; MLX is never imported on Pi

---

## Quick start

### 1. Clone and install locally

```bash
git clone https://github.com/YuvrajSingh-mist/smoltorrent
cd smoltorrent
uv sync
```

### 2. Configure your cluster

Edit `configs/config.yaml`:

```yaml
data_path: test/fixtures/mlx-community--SmolLM2-135M-Instruct/model.safetensors
save_path: ~/Desktop/smoltorrent/received_model/model.safetensors

num_workers: 4

devices_config:
  master:
    - host: localhost
      ip: 100.78.120.114
      rank: 0
      port: 5000
  workers:
    - host: pi4-1
      ip: 100.68.124.90
      rank: 1
      port: 5001
    - host: pi4-2
      ip: 100.79.150.107
      rank: 2
      port: 5002
    - host: pi4-3
      ip: 100.105.164.35
      rank: 3
      port: 5003
    - host: pi4-4
      ip: 100.77.162.23
      rank: 4
      port: 8004
```

### 3. Set up SSH access

```bash
ssh-keygen -t ed25519 -f ~/.ssh/smoltorrent_key
ssh-copy-id -i ~/.ssh/smoltorrent_key.pub pi@pi4-1
# Repeat for each worker
```

### 4. Launch

```bash
bash scripts/launch.sh
```

The launcher automatically:

1. Rsyncs the codebase to every node
2. Installs `uv`, creates `.venv`, runs `uv sync` on every node
3. Kills any stale tmux sessions from a previous run
4. Starts the REST API on the master
5. Starts one worker process per node with the correct rank and hostname

---

## Usage

### Dry run (prints what would happen, no SSH/launch)

```bash
bash scripts/launch.sh --dry-run
```

### Distribute model shards to workers

```bash
curl -X POST "http://localhost:8000/store-shard?model_id=mlx-community/SmolLM2-135M-Instruct"
```

### Trigger shard gather manually

`main.py` runs three checks before calling the API:
1. **Heartbeat** — TCP ping every worker; aborts if any are unreachable
2. **Shard count** — SSHes to each worker and counts `.safetensors` files on disk
3. **Gather** — only proceeds if all shards are present

```bash
uv run main.py --model-id mlx-community/SmolLM2-135M-Instruct
# Checking heartbeats... all alive
# Checking shards... 4/4 present
# Gathered 4 shards -> ~/Desktop/smoltorrent/received_model/model.safetensors
```

### Monitor sessions

```bash
# API logs (on master)
tmux attach -t syncps_api

# Worker logs (SSH into a Pi first)
tmux attach -t syncps_worker_1

# All cluster logs
tail -f logging/cluster-logs/*.log
```

---

## Project layout

```
smoltorrent/
├── algorithms/
│   └── SyncPS/
│       └── worker.py           # TCP listener: store_shard (write to disk) + send_shard (read from disk)
├── backend/
│   ├── api.py                  # FastAPI REST API (port 8000) — /store-shard, /gather-shards
│   └── README.md               # API endpoint reference
├── networking/
│   └── send_receive.py         # Length-prefixed TCP messaging with bandwidth metrics
├── utils/
│   ├── check_workers.py        # TCP heartbeat check against all configured workers
│   ├── common_utils.py         # shard_to_bytes(), shard_from_bytes(), chunk_data(), save_received_data_shard()
│   ├── log_utils.py            # Coloured per-component cluster logging
│   └── network_metrics.py      # Send/recv bandwidth and latency tracking
├── scripts/
│   └── launch.sh               # Full cluster orchestrator (rsync -> deps -> cleanup -> launch)
├── test/
│   ├── README.md               # Test marker reference and run commands
│   ├── test_dir_name_conversion.py           # Unit: model_id_to_dir_name()
│   ├── test_chunk_data.py                    # Unit: chunk_data() tensor sharding
│   ├── test_cli_args_and_shard_count.py      # Unit + SSH: main.py CLI, _count_remote_shards
│   ├── test_api.py                           # API: /gather-shards and /store-shard
│   ├── test_gather_shards_to_master.py       # Integration: gather -> merge -> infer
│   ├── test_shard_store_and_gather.py        # Integration: shard round-trip via common_utils
│   └── test_smollm2.py                       # Smoke: load fixture model, run MLX inference
├── configs/
│   └── config.yaml             # Cluster topology, model paths, worker count
├── main.py                     # CLI: heartbeat -> shard count -> POST /gather-shards
└── pyproject.toml
```

---

## Configuration reference

| Key | Description |
|---|---|
| `data_path` | Source model weights (`.safetensors`) to distribute |
| `save_path` | Where the reassembled model is written on master |
| `n_chunks` | Number of shards to split the model into |
| `num_workers` | Expected number of worker connections |
| `devices_config.master` | Master host, IP, rank (always 0), port |
| `devices_config.workers` | Per-worker: host, IP, rank (1…N), port |
| `log_dir` | *(optional)* Override log output directory (default: `/tmp/smolcluster-logs`) |

---

## Shard storage

Each shard is saved on the Pi worker that holds it:

```
shards/incoming_shards/
  {model_name}/
    worker-{rank}/
      {model_name}_shard_{rank}.safetensors
      {model_name}_shard_{rank}.safetensors.metadata.json
```

`model_name` is the HF model ID with `/` replaced by `--` (e.g. `mlx-community--SmolLM2-135M-Instruct`). The sidecar `.metadata.json` contains: `hostname`, `platform_machine`, `pid`, `saved_at_utc`, `rank`, `role`, and `config_path`.

After gather, the master also saves a local copy of each shard under the same layout (in `shards/incoming_shards/` on the master), then merges them to `save_path`.

---

## License

See [LICENSE](LICENSE).
