# backend

FastAPI REST API that runs on the master node (port 8000).

## Endpoints

### `POST /store-shard`

Loads the model from `data_path` in config, splits it into `N` shards (one per worker), and sends each shard to its ranked Pi worker over TCP. Each shard is checksummed (SHA-256) before sending; the worker verifies the checksum and acks with the path it wrote to. Failed sends are retried with exponential backoff (`2^attempt` seconds, up to `MAX_RETRIES=3`) on a background daemon thread — other workers are not blocked.

No file upload needed — the model is already on disk at the master.

| Query param | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | No | HuggingFace-style ID (e.g. `mlx-community/SmolLM2-135M-Instruct`). Slashes become `--` for directory naming. Falls back to `data_path` parent name in config if omitted. |

**200 OK** — `text/plain` streaming. One log line per event, flushed as each shard lands:

```
Loaded 148 tensors (270.4 MB) — chunking into 4 shards
  ✓ rank 1 (pi4-1)
  ✓ rank 2 (pi4-2)
  ✓ rank 3 (pi4-3)
  ✓ rank 4 (pi4-4)
Done: 4/4 shards stored for mlx-community--SmolLM2-135M-Instruct
```

On partial failure, failed ranks emit `  ✗ rank N (host) permanently failed: <reason>` and the final line is `ERROR: N/M shards failed`.

**404** — `data_path` from config does not exist on disk (emitted as `ERROR:` line in the stream).

---

### `POST /gather-shards`

Connects to every configured worker via TCP, requests each worker's shard (`send_shard` command), receives the serialized bytes, deserializes them, merges all shards into one model, and writes the result to `save_path`. Workers load their shard from disk on demand — the shard does not need to be in memory from a previous store.

| Query param | Type | Required | Description |
|---|---|---|---|
| `model_id` | string | No | HuggingFace-style ID. Falls back to `data_path` parent name in config if omitted. |

**200 OK** — `text/plain` streaming. One log line per event, flushed as each shard arrives and is saved:

```
  ✓ rank 1 (pi4-1) — saved → shards/incoming_shards/.../worker-1/model_shard_1.safetensors
  ✓ rank 2 (pi4-2) — saved → ...
  ...
Merging 4 shards → ~/Desktop/smoltorrent/received_model/model.safetensors
Done: saved → ~/Desktop/smoltorrent/received_model/model.safetensors
```

Each shard is written to disk immediately as it arrives rather than buffering all shards before saving. On partial failure the final line is `ERROR: N/M shards failed — skipping merge`.

---

## Wire format

Shards are sent as raw `safetensors` bytes over TCP. This is the cross-platform format:

- **Master (macOS/MLX)** — `shard_to_bytes`: converts MLX arrays to torch tensors via raw bit reinterpretation (`torch.frombuffer`), then serializes with `safetensors.torch.save()`. `shard_from_bytes`: deserializes with `mx.load()` returning MLX arrays.
- **Worker (Pi/torch)** — `shard_from_bytes`: `safetensors.torch.load()` returns torch tensors directly. `shard_to_bytes`: `safetensors.torch.save()` directly.

The safetensors format stores raw bytes + dtype string + shape with no framework dependency, so the same bytes are readable by both MLX and torch.

---

## Shard storage layout

Shards pulled by `/gather-shards` are cached locally on the master under:

```
shards/incoming_shards/
  {model_name}/
    worker-{rank}/
      {model_name}_shard_{rank}.safetensors
      {model_name}_shard_{rank}.safetensors.metadata.json
```

Shards stored by workers (via `/store-shard`) follow the same layout, but on the Pi's filesystem.

---

## Running standalone

```bash
# From project root
uv run uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Or via the cluster launcher (recommended):

```bash
bash scripts/launch.sh            # starts API + all workers
bash scripts/launch.sh --api-only # starts API only (workers must already be running)
```
