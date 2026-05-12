# backend

FastAPI REST API that runs on the master node (port 8000). Workers keep their shards in memory; this API pulls them over TCP and assembles the final model.

## Endpoints

### `POST /gather-shards`

Connects to every configured worker, pulls each shard via the TCP socket protocol, merges them, and writes the reassembled model to `save_path`.

| Query param | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | No | HuggingFace-style ID (e.g. `mlx-community/Qwen2.5-0.5B-Instruct-bf16`). Slashes are converted to `--` for directory naming. Falls back to `data_path` in config if omitted. |

**200 OK**
```json
{
  "gathered": [
    {"rank": 1, "host": "pi4-1", "shard_path": "shards/incoming_shards/.../worker-1/...safetensors"},
    ...
  ],
  "save_path": "~/Desktop/smoltorrent/received_model/model.safetensors"
}
```

**500** — one or more workers failed. Body includes `gathered` (successes so far) and `errors` (per-rank failure reasons).

---

### `POST /store-shard`

Accepts a `.safetensors` file upload and saves it to the shard store. Used when a worker pushes its shard to master directly (multipart form data).

| Form field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | required | `.safetensors` shard file |
| `rank` | int | required | Worker rank |
| `role` | string | No | Label stored in metadata (default: `"received"`) |
| `host` | string | No | Source hostname stored in metadata |
| `output_dir` | string | No | Override destination directory |

**200 OK**
```json
{"shard_path": "...", "metadata_path": "...", "rank": 2}
```

---

## Shard storage layout

Incoming shards land under:

```
shards/incoming_shards/
  {model_name}/
    worker-{rank}/
      {model_name}_shard_{rank}.safetensors
      {model_name}_shard_{rank}.safetensors.metadata.json
```

`model_name` is the HF model ID with `/` replaced by `--` (e.g. `mlx-community--SmolLM2-135M-Instruct`).

---

## Running standalone

```bash
# From project root
uv run uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Or via the cluster launcher (recommended):

```bash
bash scripts/launch.sh          # starts API + server + workers
bash scripts/launch.sh --api-only   # starts API only (workers must already be running)
```
