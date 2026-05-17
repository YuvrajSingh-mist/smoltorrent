# backend

FastAPI REST API that runs on the master node (port 8000). Launched via `bash scripts/launch.sh`.

## Endpoints

### `POST /store-shard`

Loads a checkpoint from disk, splits tensors evenly into N shards (one per worker), computes a SHA-256 checksum per shard, and sends each over TCP to its ranked Pi worker. Workers verify the checksum and write the shard + `.checksum` sidecar to disk.

| Query param | Type | Required | Description |
|---|---|---|---|
| `ckpt_path` | string | Yes | Absolute path to the `.safetensors` checkpoint on the master |

**200 OK** — `text/plain` streaming, one log line per event:

```
Loaded 290 tensors (942.3 MB) from Qwen2.5-0.5B/gaming/latest — chunking into 4 shards
  ✓ rank 1 (pi4-1)
  ✓ rank 2 (pi4-2)
  ✓ rank 3 (pi4-3)
  ✓ rank 4 (pi4-4)
Done: 4/4 shards stored → Qwen2.5-0.5B/gaming/latest
```

On partial failure: failed ranks emit `↻ rank N (host) failed — queuing retry: <reason>`. Retried with exponential backoff (`2^attempt` seconds, up to `MAX_RETRIES=6`). Permanently failed shards emit `✗ rank N permanently failed` and the final line is `ERROR: N/M shards failed`.

---

### `POST /gather-shards`

Connects to every configured worker via TCP, pulls each shard, saves it locally, then merges all shards into one `.safetensors` file written as `merged.safetensors` next to the original checkpoint.

| Query param | Type | Required | Description |
|---|---|---|---|
| `ckpt_path` | string | Yes | Absolute path to the original checkpoint (same path used for store) |

**200 OK** — `text/plain` streaming:

```
  ✓ rank 1 (pi4-1) — saved → shards/worker_1/Qwen2.5-0.5B/gaming/latest/shard.safetensors
  ✓ rank 2 (pi4-2) — saved → ...
  ...
Merging 4 shards → ~/smolcluster/checkpoints/Qwen2.5-0.5B/gaming/latest/merged.safetensors
Done: saved → ~/smolcluster/checkpoints/Qwen2.5-0.5B/gaming/latest/merged.safetensors
```

Each shard is saved to disk as it arrives — a mid-gather failure doesn't lose already-received shards. On partial failure: `ERROR: N/M shards failed — skipping merge`.

---

## Wire format

Shards travel as raw `safetensors` bytes over TCP.

| Side | serialize (`shard_to_bytes`) | deserialize (`shard_from_bytes`) |
|---|---|---|
| Master (macOS/MLX) | MLX arrays → `torch.frombuffer` reinterpret → `safetensors.torch.save()` | `mx.load()` → MLX arrays |
| Worker (Pi/torch) | `safetensors.torch.save()` directly | `safetensors.torch.load()` → torch tensors |

Safetensors is the only format that carries shape + dtype + tensor name cleanly across frameworks with no framework-specific code embedded in the bytes.

**Important:** on gather, the master receives shard bytes and deserializes to MLX arrays (Server). Saving must use `_save_shard()` (which calls `mx.save_safetensors`) — using `safetensors.torch.save_file()` directly will crash because it expects torch tensors, not MLX arrays.

---

## Shard storage layout

On each Pi worker:
```
~/Desktop/smoltorrent/shards/worker_{rank}/
  {model}/{experiment}/{run}/latest/
    shard.safetensors
    shard.checksum        ← SHA-256 of shard file, used by checksum_sync
```

Local cache on master after gather:
```
~/smoltorrent/shards/worker_{rank}/
  {model}/{experiment}/{run}/latest/
    shard.safetensors
```

Merged output:
```
~/smolcluster/checkpoints/{model}/{experiment}/{run}/latest/
  merged.safetensors
```

---

## Running standalone

```bash
uv run uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Or via the cluster launcher (recommended):

```bash
bash scripts/launch.sh            # starts API + watcher + all workers
bash scripts/launch.sh --api-only # starts API only (workers must already be running)
```
