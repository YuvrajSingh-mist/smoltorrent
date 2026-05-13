# Tests

## Structure

```
test/
├── fixtures/                               # Read-only model weights (never written to)
│   └── mlx-community--SmolLM2-135M-Instruct/
├── test_dir_name_conversion.py             # Unit: model_id_to_dir_name()
├── test_chunk_data.py                      # Unit: chunk_data() tensor sharding logic
├── test_cli_args_and_shard_count.py        # Unit + SSH: main.py CLI args, _count_remote_shards
├── test_api.py                             # API: /gather-shards and /store-shard endpoints
├── test_gather_shards_to_master.py         # Integration: gather -> merge -> inference
├── test_shard_store_and_gather.py          # Integration: shard round-trip via common_utils
└── test_smollm2.py                         # Smoke: load fixture model and run MLX inference
```

## Markers

| Marker        | Meaning                                                                      | Runs by default |
|---------------|------------------------------------------------------------------------------|-----------------|
| *(none)*      | Pure unit tests, no I/O                                                      | yes             |
| `integration` | Requires model weights in `test/fixtures/`                                   | yes             |
| `ssh`         | SSHes into live Pi workers using `~/.ssh/config` entries from `config.yaml`  | yes             |
| `api`         | Hits the real FastAPI server (`bash scripts/launch.sh` must be running)      | no              |

`addopts = "-m 'not api'"` in `pyproject.toml` — ssh tests run by default.

## Running

```bash
# Default (unit + integration + ssh)
uv run pytest test/

# Unit and integration only (no cluster needed)
uv run pytest test/ -m "not ssh and not api"

# SSH tests only — requires cluster running
uv run pytest test/test_cli_args_and_shard_count.py -m ssh -v

# API tests — requires server running (bash scripts/launch.sh)
uv run pytest test/test_api.py -m api -v
```

## Write Paths

Tests never write inside `test/fixtures/`. All generated artifacts go to:

| Artifact | Path |
|---|---|
| Incoming shards (test) | `shards/incoming_shards/` |
| Merged weights (temp) | `shards/merged.safetensors` |
| Reconstructed model | `received_model/` |
