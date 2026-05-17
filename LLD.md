# SmolTorrent — Low-Level Design

---

## Wire protocol

All TCP communication uses a 4-byte big-endian length prefix followed by a pickle-serialized payload.

```
┌──────────────┬───────────────────────────┐
│  4 bytes     │  N bytes                  │
│  big-endian  │  pickle.dumps(message)    │
│  length = N  │                           │
└──────────────┴───────────────────────────┘
```

The receiver pre-allocates a `bytearray` of the exact message length and uses `recv_into` with a `memoryview` — zero copies, zero allocations during receive. (The old `data += chunk` pattern on immutable `bytes` caused O(n²) copying: ~240 GB of memcpy for a 169 MB shard, turning a 2-minute transfer into 13 minutes on Pi's SD card.)

---

## Command protocol (`algorithms/SyncPS/worker.py`)

The master sends a tuple; the worker replies with a tuple. One connection per command.

### `heartbeat`
```
master → worker:  ("heartbeat",)
worker → master:  "alive"
```
Health check. Called before operations and by `launch.sh` to verify workers are up.

---

### `store_shard`
```
master → worker:  ("store_shard", rank, shard_bytes, checksum, rel_path)
worker → master:  ("store_shard_done", rank, shard_path)
               or ("store_shard_failed", rank, error_msg)
```
Worker:
1. Verifies `SHA-256(shard_bytes) == checksum` — rejects on mismatch
2. Deserializes `shard_bytes` via `shard_from_bytes` → torch tensors
3. Writes `shards/worker_{rank}/{rel_path}/shard.safetensors`
4. Computes SHA-256 of the written file, writes `shard.checksum` sidecar
5. Replies done or failed

Why the sidecar: `store_shard` verifies the bytes in transit. The sidecar is for the startup `checksum_sync` sweep — it lets the worker detect disk corruption between runs without the master resending the original bytes.

---

### `send_shard`
```
master → worker:  ("send_shard", rank, rel_path)
worker → master:  shard_bytes  (raw safetensors bytes)
               or None         (shard not found)
```
Worker reads `shards/worker_{rank}/{rel_path}/shard.safetensors` and sends the raw bytes. The master deserializes with `shard_from_bytes` (→ MLX on Mac, torch on Pi).

---

### `sync`
```
master → worker:  ("sync", rank, extensions)
worker → master:  [rel_path, rel_path, ...]
```
Worker globs `shards/worker_{rank}/` for all `shard.safetensors` files and returns their parent dirs relative to the worker root. The `extensions` parameter is passed for protocol compatibility but ignored — shards are always stored as `shard.safetensors` regardless of the source checkpoint extension.

The master calls this against all workers in parallel and takes the **intersection** — only paths present on every worker. This is the "what we already have" baseline before deciding what to transfer.

---

### `checksum_sync`
```
master → worker:  ("checksum_sync", rank, rel_path)
worker → master:  ("checksum_sync_result", "ok",           rel_path)
               or ("checksum_sync_result", "mismatch",     rel_path)
               or ("checksum_sync_result", "shard_missing", rel_path)
```
Worker:
1. If `shard.checksum` doesn't exist → bootstrap it by hashing the shard now, reply `"ok"`
2. Otherwise: hash the shard, compare to stored checksum → reply `"ok"` or `"mismatch"`

Called only on the **startup** trigger. All paths passed to `checksum_sync` are from the intersection (all workers confirmed having them via `sync`), so `"shard_missing"` is a defensive response only.

---

### `all_shards_present`
```
master → worker:  ("all_shards_present", rank, [rel_path, ...])
worker → master:  [missing_rel_path, ...]   (empty list = all present)
```
Worker checks whether `shards/worker_{rank}/{rel_path}/shard.safetensors` exists for each path in the list, returns the missing ones.

Called after every transfer batch (crosscheck phase). Unlike `sync` which returns an intersection, this queries each worker individually — so it can report *which* worker is missing *which* paths, not just whether everyone has everything.

---

## Serialization path (`utils/common_utils.py`)

### Mac → bytes (`shard_to_bytes`)

MLX arrays can't be passed to `safetensors.torch.save` directly (different framework).

```python
bytes(mlx_array)                          # raw memory via buffer protocol — bfloat16 safe
→ torch.frombuffer(bytearray(...), dtype) # reinterpret bits as torch tensor — no numpy
→ safetensors.torch.save(torch_dict)      # → bytes
```

This is the only path that works for bfloat16 — numpy's PEP 3118 interpreter raises an item-size mismatch on bfloat16.

### bytes → tensors (`shard_from_bytes`)

```python
# Mac
mx.load(_NamedBytesIO(data))   # MLX needs file-like with .name ending in .safetensors
# Pi
safetensors.torch.load(data)   # returns torch tensors directly
```

`_NamedBytesIO` is a `BytesIO` subclass with `name = "shard.safetensors"` hardcoded — MLX's `load` inspects the `.name` attribute to determine format.

---

## Chunking (`utils/common_utils.py` → `chunk_data`)

```python
idx = torch.tensor(list(range(len(data))))
chunked_tensors = torch.chunk(idx, n_chunks)   # splits index tensor evenly
```

Worker `i` always gets the same subset of keys for the same checkpoint — deterministic, reproducible. `torch.chunk` returns at most `n_chunks` chunks; for models with many layers (real LLMs) this is always exactly `n_chunks`.

---

## Store endpoint (`backend/api.py` → `/store-shard`)

```
load_tensors(ckpt_path)
chunk_data(tensors, n_workers)
for each chunk:
    shard_bytes = shard_to_bytes(chunk)     # serialize once
    checksum    = compute_checksum(shard_bytes)

ThreadPoolExecutor(max_workers=n_workers):
    for each worker: _send_shard_to_worker(worker, shard_bytes, checksum, rel_path)
        → ("store_shard", rank, shard_bytes, checksum, rel_path)
        ← ("store_shard_done", ...) or failure

failed sends → retry_queue (daemon thread, 2^attempt sec backoff, 6 max retries)
store_queue.join()    ← single wait point
stream log lines as text/plain
```

Why serialize before spawning threads: serializing inside the thread would race on the MLX array (not thread-safe). Serializing all shards up front also means if shard 3 fails to serialize, shards 1 and 2 were never sent — cleaner error state.

Why streaming response: transfers take minutes. A non-streaming response would timeout at the client. One log line per event keeps the connection alive and gives the caller progress.

---

## Gather endpoint (`backend/api.py` → `/gather-shards`)

```
rel_path = ckpt_path.parent.relative_to(ckpt_root)

for each worker (parallel):
    → ("send_shard", rank, rel_path)
    ← shard_bytes
    shard_from_bytes(shard_bytes)
    save to SHARDS_ROOT/worker_{rank}/{rel_path}/shard.safetensors  ← on arrival

gather_queue.join()
merge_shards(shards_by_rank.values())
save_merged_model → ckpt_root/{rel_path}/merged.safetensors
```

Save-on-arrival (not buffer-then-save): a mid-gather failure previously discarded all shards received so far. Now each shard hits disk as soon as it arrives.

---

## Retry mechanism (`backend/api.py` → `_retry_worker`)

```python
while True:
    item = retry_queue.get()        # blocks until work appears
    if item["attempt"] > MAX_RETRIES:
        dead_letter.append(...)
        retry_queue.task_done()
        continue
    time.sleep(2 ** item["attempt"])
    ok, err, result = send_fn(...)
    if ok:
        recovered.append(result)
    else:
        retry_queue.put({..., "attempt": attempt + 1})
    retry_queue.task_done()
```

The retry thread is a daemon — it doesn't block the main loop. `queue.join()` is the single synchronisation point: it blocks until all tasks have called `task_done()`, whether via success, retry success, or dead-letter.

---

## Watcher transfer loop (`watcher/watch.py`)

```
startup = True
while True:
    trigger.wait()      # blocks until threading.Event is set
    trigger.clear()

    # Phase 1: file_sync
    worker_paths = _sync_all_workers(workers, extensions)   # parallel sync, takes intersection
    local_paths  = _scan_local(ckpt_root, extensions)
    to_transfer  = [p for p in local_paths if rel(p) not in worker_paths]

    # Phase 2: checksum_sync (startup only)
    if startup:
        corrupted = _checksum_sync_all(workers, intersection, ckpt_root)
        checksum_retry = [p for p in intersection if rel(p) in corrupted]
        startup = False

    # Phase 3: transfer
    for path in checksum_retry + to_transfer:
        request_store_shards(ckpt_path=str(path), log_fn=logger.info)

    # Phase 4: crosscheck
    crosscheck_retry = _crosscheck_all_workers(workers, local_paths, ckpt_root)
    for path in crosscheck_retry:
        request_store_shards(ckpt_path=str(path), log_fn=logger.info)

    # Re-evaluate pending (files that were unstable when detected)
    now_stable = [p for p in pending if _is_stable(p)]
    if now_stable:
        trigger.set()   # re-trigger for the newly stable files
```

### Why intersection for phase 1, not union

The intersection tells the master what it can skip. If a path is in the intersection, all workers have it — no transfer needed. If it's not in the intersection, at least one worker is missing it — transfer it. The crosscheck in phase 4 then confirms every worker received it.

### Why checksum_sync runs in parallel per file (not per worker)

`_checksum_sync_all` submits tasks as `for p in intersection for w in workers` (outer = files, inner = workers). With 4 workers and 4 threads, the first 4 tasks are one per worker — all workers hash concurrently. If the loop were `for w in workers for p in intersection`, all tasks for worker 1 would be submitted first, all 4 threads would pile on worker 1 serially, then move to worker 2 — no parallelism across workers.

---

## Shard on-disk layout

```
Pi:
~/Desktop/smoltorrent/shards/worker_{rank}/{rel_path}/
    shard.safetensors
    shard.checksum          ← SHA-256 of shard.safetensors

Mac (local cache after gather):
~/smoltorrent/shards/worker_{rank}/{rel_path}/
    shard.safetensors

Mac (merged output):
~/smolcluster/checkpoints/{rel_path}/
    model.safetensors       ← original training checkpoint
    merged.safetensors      ← reassembled from Pi shards
```

`rel_path` = `ckpt_file.parent.relative_to(ckpt_root)`, e.g. `Qwen2.5/run1/latest`. It's the same on both Mac and Pi — the master uses it as the storage key, and the worker uses it as the directory name under its `shards/worker_{rank}/` root.

---

## Auto-start

### Mac (`/Library/LaunchDaemons/`)

The plist runs `/usr/local/bin/smoltorrent_startup.sh` at boot (before login). The script pings a Pi every 5s until Tailscale is up (5 min timeout), then runs `launch.sh`.

Key plist fields:
- `UserName` — runs as the user, not root (required to access home directory)
- `EnvironmentVariables.PATH` — includes `/opt/homebrew/bin` so `uv` and `tmux` are found
- `KeepAlive = false` — runs once at boot, exits; cluster keeps running in tmux independently

### Pi (`systemd` template unit)

```ini
[Unit]
After=network-online.target tailscaled.service

[Service]
ExecStart=/path/to/.venv/bin/python algorithms/SyncPS/worker.py %i $(hostname)
Restart=on-failure
RestartSec=5
```

`%i` is the rank, injected by systemd when starting `smoltorrent-worker@1.service`. One unit file covers all ranks. `Restart=on-failure` brings the worker back automatically after a crash.
