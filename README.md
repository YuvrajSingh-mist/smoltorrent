# smoltorrent

Distributed inference and fine-tuning across heterogeneous edge devices using a synchronous parameter server (SyncPS). Built to run on Raspberry Pi clusters, Linux x86 nodes, and macOS — coordinated from a single `launch.sh`.

```
Master (macOS)
  ├── Parameter Server   (algorithms/SyncPS/server.py)
  ├── REST API           (backend/api.py)
  └── Workers × N        (algorithms/SyncPS/worker.py  ← runs on each Pi)
```

---

## How it works

1. **Server** loads a `.safetensors` model, splits it into `N` shards (one per worker), and listens for connections.
2. **Workers** (each on a Pi) register with the server, receive their shard, store it locally.
3. Once all `N` shards are received, the server runs an integration test that:
   - Gathers all shards back to master
   - Merges them into a single model
   - Runs inference to verify the reassembled model works end-to-end
4. `main.py` provides a one-line CLI to trigger shard collection manually via the REST API.

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
- `torch[cpu]` is used everywhere — no CUDA required
- `mlx-lm` is installed **only on macOS** (for inference verification on master)
- Workers (Raspberry Pi / Linux) never touch MLX

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
save_path: ~/Desktop/received_model/mlx-community--SmolLM2-135M-Instruct/model.safetensors

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

Use `host: localhost` for the master if the orchestrator runs on the same machine as the server. Worker `host` values must match the hostnames you SSH into.

### 3. Set up SSH access

```bash
ssh-keygen -t ed25519 -f ~/.ssh/smoltorrent_key
ssh-copy-id -i ~/.ssh/smoltorrent_key.pub ubuntu@pi4-1
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
4. Starts the REST API and parameter server on master
5. Starts one worker process per node with the correct rank and hostname

---

## Usage

### Dry run (prints what would happen, no SSH/launch)

```bash
bash scripts/launch.sh --dry-run
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
# Server logs (on master)
tmux attach -t syncps_server

# Worker logs (on master or any node)
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
│       ├── server.py           # Parameter server: loads model, shards it, accepts workers
│       └── worker.py           # Worker: registers, receives shard, stores locally; handles heartbeat
├── backend/
│   ├── api.py                  # FastAPI REST API (port 8000) — /gather-shards, /store-shard
│   └── README.md               # API endpoint reference
├── networking/
│   └── send_receive.py         # Length-prefixed TCP messaging with bandwidth metrics
├── utils/
│   ├── check_workers.py        # TCP heartbeat check against all configured workers
│   ├── common_utils.py         # chunk_data(), save_received_data_shard(), model_id_to_dir_name()
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
| `devices_config.master` | Server host, IP, rank (always 0), port |
| `devices_config.workers` | Per-worker: host, IP, rank (1…N), port |
| `received_shards_dir` | *(optional)* Override incoming shard storage path (default: `shards/incoming_shards`) |
| `log_dir` | *(optional)* Override log output directory (default: `/tmp/smolcluster-logs`) |

---

## Shard storage

Each received shard is saved as:

```
shards/incoming_shards/
  {model_name}/
    worker-{rank}/
      {model_name}_shard_{rank}.safetensors
      {model_name}_shard_{rank}.safetensors.metadata.json
```

`model_name` is the HF model ID with `/` replaced by `--` (e.g. `mlx-community--SmolLM2-135M-Instruct`). The sidecar `.metadata.json` contains: `hostname`, `platform_machine`, `pid`, `saved_at_utc`, `rank`, `role`, and `config_path`.

---

## Integration test

After all shards are received, the server automatically runs:

```bash
pytest test/test_gather_shards_to_master.py -v -s -m integration
```

This test:
1. Gathers all per-node shards from disk
2. Merges them with no key overlap (raises `ValueError` if any tensor appears in two shards)
3. Reassembles the full model directory
4. Runs MLX inference on the master with the prompt: *"Explain what a BitTorrent tracker does in one short paragraph."*
5. Asserts the response is non-empty and prints it

---

## License

See [LICENSE](LICENSE).

