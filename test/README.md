# Tests

## Structure

```
test/
├── fixtures/                                     # Read-only model weights (never written to)
│   └── mlx-community--SmolLM2-135M-Instruct/
├── test_dir_name_conversion.py                   # Unit: model_id_to_dir_name()
├── test_chunk_data.py                            # Unit: chunk_data() tensor sharding logic
├── test_cli_args_and_shard_count.py              # Unit + SSH: main.py CLI args
├── test_worker_commands.py                       # Integration: all worker TCP commands (heartbeat/sync/store/send/checksum)
├── test_watcher_logic.py                         # Integration: watcher sync, crosscheck, file trigger, extension filter
├── test_pending_loop.py                          # Integration: pending loop — real file sizes (~150–400 MB), real worker APIs
├── test_api.py                                   # API: /gather-shards and /store-shard endpoints
├── test_gather_shards_to_master.py               # Integration: gather -> merge
├── test_shard_store_and_gather.py                # Integration: shard round-trip via common_utils
├── test_received_model_inference.py              # Integration: load gathered Qwen2.5 weights, run MLX inference
└── test_smollm2.py                               # Smoke: load fixture model and run MLX inference
```

## Markers

| Marker | Meaning | Runs by default |
|---|---|---|
| *(none)* | Pure unit tests, no I/O | yes |
| `integration` | Requires live Pi workers via `configs/config.yaml` | yes |
| `ssh` | SSHes into live Pi workers using `configs/config.yaml` | yes |
| `api` | Hits the real FastAPI server on port 8000 (cluster must be running) | no |

`addopts = "-m 'not api'"` in `pyproject.toml` — `api` tests are excluded by default.

## Running

```bash
# Default (unit + integration + ssh, no api)
uv run pytest test/

# Unit only — no cluster, no network
uv run pytest test/ -m "not ssh and not api and not integration"

# Unit and integration only (no cluster needed)
uv run pytest test/ -m "not ssh and not api"

# Worker TCP commands (all 6 commands: heartbeat/sync/store/send/checksum/all_shards_present)
uv run pytest test/test_worker_commands.py -v -m integration

# Watcher logic (sync, crosscheck, file trigger, extension filter)
uv run pytest test/test_watcher_logic.py -v -m integration

# Pending loop (real file sizes, real worker APIs — takes ~5 min)
uv run pytest test/test_pending_loop.py -v -m integration

# Inference test against gathered Qwen2.5 gaming checkpoint
uv run pytest test/test_received_model_inference.py -v -m integration -s

# API tests — requires cluster running (bash scripts/launch.sh)
uv run pytest test/test_api.py -m api -v
```

## Viewing generation output

Pytest captures stdout by default. Pass `-s` to see MLX generation output live:

```bash
uv run pytest test/test_received_model_inference.py -s -k test_generate
```

## Write paths

Tests never write inside `test/fixtures/`. All generated artifacts go to:

| Artifact | Path |
|---|---|
| Gathered + merged weights | `~/smolcluster/checkpoints/{model}/{run}/latest/merged.safetensors` |
| Local shard cache (after gather) | `shards/worker_{rank}/{model}/{run}/latest/shard.safetensors` |
