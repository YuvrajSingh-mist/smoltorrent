# backend

FastAPI REST API that runs on the master node (port 8000). Launched via `bash scripts/launch.sh`.

## Endpoints

### `POST /store-shard`

Loads a checkpoint from disk, splits tensors evenly into N shards (one per worker), serializes all shards upfront, then fires all N×REDUNDANCY sends simultaneously in one thread pool. Each send carries a SHA-256 checksum; workers verify and write the shard + `.checksum` sidecar to disk.

| Query param | Type | Required | Description |
|---|---|---|---|
| `ckpt_path` | string | Yes | Absolute path to the `.safetensors` checkpoint on the master |

**200 OK** — `text/plain` streaming, one log line per event:

```
Loaded 290 tensors (942.3 MB) from Qwen2.5-0.5B/gaming/latest — chunking into 4 shards
  ✓ rank 1 (pi4-1) [round 0]
  ✓ rank 2 (pi4-2) [round 0]
  ✓ rank 3 (pi4-3) [round 0]
  ✓ rank 4 (pi4-4) [round 0]
  ✓ rank 2 (pi4-2) [round 1]
  ✓ rank 3 (pi4-3) [round 1]
  ✓ rank 4 (pi4-4) [round 1]
  ✓ rank 1 (pi4-1) [round 1]
Done: 8/8 sends (2x replicated) → Qwen2.5-0.5B/gaming/latest
```

Round 0 = primary (shard i → workers[i]). Round 1 = replica (shard i → workers[(i+1) % N]). Both rounds fire in parallel — order of completion is non-deterministic.

On partial failure: failed ranks emit `↻ rank N (host) failed — queuing retry: <reason>`. Retried with exponential backoff (`2^attempt` seconds, up to `MAX_RETRIES=6`). Permanently failed shards emit `✗ rank N permanently failed` and the final line is `ERROR: N/M sends failed`.

### `POST /gather-shards`

Connects to all workers in parallel via TCP, pulls each shard simultaneously (falling back to the replica worker if the primary is unreachable), saves each shard on arrival, then merges all shards into one `.safetensors` file written as `merged.safetensors` next to the original checkpoint.

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

## Wire format

Shards travel as raw `safetensors` bytes over TCP.

| Side | serialize (`shard_to_bytes`) | deserialize (`shard_from_bytes`) |
|---|---|---|
| Master (macOS/MLX) | MLX arrays → `torch.frombuffer` reinterpret → `safetensors.torch.save()` | `mx.load()` → MLX arrays |
| Worker (Pi/torch) | `safetensors.torch.save()` directly | `safetensors.torch.load()` → torch tensors |

Safetensors is the only format that carries shape + dtype + tensor name cleanly across frameworks with no framework-specific code embedded in the bytes.

**Important:** on gather, the master receives shard bytes and deserializes to MLX arrays (Server). Saving must use `_save_shard()` (which calls `mx.save_safetensors`) — using `safetensors.torch.save_file()` directly will crash because it expects torch tensors, not MLX arrays.

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

## Running standalone

```bash
uv run uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Or via the cluster launchers:

```bash
# SSH setup (production) — rsyncs code, starts API + watcher + workers on all Pis
bash scripts/launch.sh
bash scripts/launch.sh --api-only  # API only (workers must already be running)

# grove flow (testing) — workers already running via grove join; starts API + watcher only
bash scripts/grove_launch.sh
```
