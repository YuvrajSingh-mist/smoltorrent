# Tests

## Structure

```
test/
├── fixtures/                          # Read-only model weights (never written to)
│   └── mlx-community--SmolLM2-135M-Instruct/
├── test_chunk_data.py                 # Unit tests for tensor sharding logic
├── test_gather_cli.py                 # Tests for main.py (heartbeat, shard count, gather flow)
├── test_gather_shards_to_master.py    # Integration test: shard -> merge -> inference
└── test_smollm2.py                    # Smoke test: load fixture model and run inference
```

## Markers

| Marker        | Meaning                                             | Runs by default |
|---------------|-----------------------------------------------------|-----------------|
| *(none)*      | Pure unit tests, no I/O                             | yes             |
| `integration` | Requires model weights in `test/fixtures/`          | yes             |
| `ssh`         | SSHes into live Pi workers via `~/.ssh/config`      | no              |
| `api`         | Hits the real FastAPI server (`bash scripts/launch.sh` must be running) | no |

`addopts = "-m 'not ssh and not api'"` in `pyproject.toml` excludes the live-cluster markers by default.

## Running

```bash
# Default (unit + integration)
uv run pytest test/

# Only unit tests (no model weights needed)
uv run pytest test/ -m "not integration and not ssh and not api"

# SSH tests — requires cluster running
uv run pytest test/test_gather_cli.py -m ssh -v

# API tests — requires server running (bash scripts/launch.sh)
uv run pytest test/test_gather_cli.py -m api -v
```

## Write Paths

Tests never write inside `test/fixtures/`. All generated artifacts go to:

| Artifact | Path |
|---|---|
| Incoming shards (test) | `shards/incoming_shards/` |
| Merged weights (temp) | `shards/merged.safetensors` |
| Reconstructed model | `received_model/` |
