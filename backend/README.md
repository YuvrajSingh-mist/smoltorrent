# backend

FastAPI REST API that runs on the master node (port 8000).

## Endpoints

### `POST /gather-shards`

Connects to every configured worker via TCP, pulls each worker's in-memory shard, merges all shards into one model, and writes the result to `save_path`.

| Query param | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | No | HuggingFace-style ID (e.g. `mlx-community/Qwen2.5-0.5B-Instruct-bf16`). Slashes are converted to `--` for directory naming. Falls back to `data_path` parent name in config if omitted. |

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

Loads the model from `data_path` in config, splits it into `N` shards (one per worker), and sends each shard to its ranked Pi worker over TCP using the `("store_shard", rank, tensor_dict)` protocol.

No file upload needed — the model is already on disk at the master.

| Query param | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | No | HuggingFace-style ID used only for naming the response. Falls back to `data_path` parent name if omitted. |

**200 OK**
```json
{
  "model_name": "mlx-community--SmolLM2-135M-Instruct",
  "num_shards": 4,
  "sent_to": [
    {"rank": 1, "host": "pi4-1"},
    {"rank": 2, "host": "pi4-2"},
    ...
  ]
}
```

**404** — `data_path` from config does not exist on disk.

**500** — one or more workers unreachable. Body includes `sent` (successes) and `errors` (per-rank failure reasons).

---

## Shard storage layout

Shards received by `handle_worker` (Pi -> master TCP push) are saved under:

```
shards/incoming_shards/
  server/
    from-rank-{rank}/
      {model_name}_shard_{rank}.safetensors
      {model_name}_shard_{rank}.safetensors.metadata.json
```

Shards pulled by `/gather-shards` (master -> Pi TCP pull) land under:

```
shards/incoming_shards/
  {model_name}/
    worker-{rank}/
      {model_name}_shard_{rank}.safetensors
```

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
