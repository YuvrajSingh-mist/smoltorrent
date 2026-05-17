# SmolTorrent — Design Decisions

---

## Wire format — safetensors (`utils/common_utils.py`)

Server runs MLX, Pi workers run torch. Pickle fails because unpickling an MLX array on Pi tries to `import mlx` which isn't installed. Numpy fails on bfloat16 because numpy doesn't natively support it — any path through numpy's dtype interpreter raises a PEP 3118 item-size mismatch.

Safetensors is the fix: it's a flat format (shape + dtype string + raw bytes) with no framework embedded. Same bytes are readable by both `mx.load()` and `safetensors.torch.load()`.

Serialization path on Server (`shard_to_bytes`):
1. `bytes(mlx_array)` — MLX exposes raw memory via the buffer protocol directly, no numpy dtype interpretation
2. `torch.frombuffer(bytearray(...), dtype=...)` — reinterprets those bits as the correct torch dtype
3. `safetensors.torch.save(torch_dict)` — produces the bytes that go on the wire

On Pi: `safetensors.torch.load(received_bytes)` → torch tensors → saved to disk as `.safetensors`.

Safetensors is used as the serialization format for the wire, not just for disk storage — it's the only format that carries shape + dtype + tensor name cleanly across frameworks.

---

## MLX gather save bug (`backend/api.py`)

`/gather-shards` received shard bytes from a Pi and called `safetensors.torch.save_file(received_shard, path)` on the master to cache it locally. On Server, `shard_from_bytes` deserializes to MLX arrays (via `mx.load()`), not torch tensors. `safetensors.torch.save_file` expects torch tensors — it crashed with `Key X is invalid, expected torch.Tensor but received mlx.core.array`.

Fix: replaced `safetensors.torch.save_file` with `_save_shard()` from `common_utils`, which branches on `_IS_MAC` and calls `mx.save_safetensors()` on macOS.

---

## Reliability — retry queue + checksum (`backend/api.py`)

A flaky worker should not block the other three or silently corrupt data.

- SHA-256 checksum computed on shard bytes before sending; worker verifies on receipt
- Workers write a `.checksum` sidecar file alongside each `shard.safetensors` for later corruption detection
- Failed sends go onto a daemon retry thread with exponential backoff (`2^attempt` seconds, up to `MAX_RETRIES=6`)
- Main loop does not wait on failures — all workers are dispatched first, retry thread runs alongside
- `store_queue.join()` is the single wait point before returning — caller always gets a definitive result
- Permanently failed shards go to `dead_letter` and are reported in the response

---

## Watcher design (`watcher/watch.py`)

The watcher monitors `ckpt_root` for new `.safetensors` files and keeps all workers in sync. Each trigger runs these phases:

1. **file_sync** — asks every reachable worker what rel_paths it has (`sync` command); takes the intersection (files present on ALL workers)
2. **checksum_sync** *(startup only)* — on the first wake-up after launch, asks every worker to SHA-256 its existing shards and compare against the `.checksum` sidecar; any mismatch is flagged for re-transfer. Skipped on all subsequent file-event triggers because per-transfer integrity is already guaranteed by the SHA-256 sent with every `/store-shard` call — re-hashing all existing files on every new file event would block new transfers for minutes.
3. **transfer** — pushes files missing from workers (not in intersection) via `/store-shard` API
4. **crosscheck** — after all transfers complete, sends `all_shards_present` to every worker with the full list of expected rel_paths; any worker that reports a missing shard triggers a re-transfer for that file

The crosscheck catches partial transfers — a file that failed mid-send on one worker but succeeded on others would appear in the intersection (missing from the intersection means all workers lacked it), but crosscheck queries each worker individually and catches the one-worker-missing case.

`_crosscheck_all_workers` accepts a `checksum=True` flag that also runs `_checksum_sync_all` on present shards and folds any corrupted paths into the re-transfer list — used during startup to combine presence check and integrity in one pass.

### Pending loop

Files detected while still being written (size changes within the 1 s stability window) are added to a `pending` list instead of triggering a transfer. A dedicated thread (`_run_pending_loop`) polls the list every 10 s, re-evaluates each file with `_is_stable`, promotes stable ones by setting `trigger`, and leaves unstable ones for the next tick.

The lock is held only for the read snapshot and the final write-back — `_is_stable` (which sleeps 1 s per file) runs outside the lock so `on_created` is never blocked waiting to append.

---

## Crosscheck command (`algorithms/SyncPS/worker.py`)

Added `all_shards_present` command: master sends `(rank, [rel_paths])`, worker checks each `shards/worker_{rank}/{rel_path}/shard.safetensors` exists, returns list of missing rel_paths. Master re-transfers anything non-empty.

---

## Double serialization bug (`backend/api.py`)

`/store-shard` serialized each shard once (to compute checksum), then passed the raw dict to `_send_shard_to_worker` which serialized it again after the socket was open. Second serialization stalled or raised mid-connection → worker saw "Socket connection broken".

Fix: `_send_shard_to_worker` takes `shard_bytes: bytes` directly. Serialized once, reused for both checksum and send.

---

## Socket timeout bug (`backend/api.py`)

`_connect_with_retry` set `sock.settimeout(2.0)` for the connect attempt and never cleared it. Every `sendall` on the returned socket had a 2s deadline. A 70 MB shard over Tailscale blows past that.

Fix: `send_message` and `receive_message` in `send_receive.py` both call `sock.settimeout(None)` as their first line — timeout is always cleared before any data transfer. No need to clear it in `_connect_with_retry`.

---

## Serialization vs pickling (`networking/send_receive.py`)

Serialization = converting an object to bytes. Deserialization = converting bytes back to the object. Pickling/unpickling is Python's specific implementation of that using the `pickle` module.

Two different serializers are used for different things:

- **Pickle** — serializes the message tuple `("store_shard", rank, shard_bytes, ...)` for TCP transport. Fine on both Server and Pi because the tuple contains only plain Python types. `shard_bytes` inside the tuple is already a `bytes` object — pickle treats it as an opaque blob.
- **Safetensors** — serializes the tensor data itself into `shard_bytes` before pickle sees it. Needed because pickling MLX arrays directly would embed the `mlx.core.array` class — unpickling on Pi would `import mlx`, which isn't installed.

Rule: pickle handles structure, safetensors handles tensors. Never let pickle see raw MLX arrays.

---

## Streaming progress (`backend/api.py`, `utils/shard_ops.py`)

Both endpoints return `text/plain` streaming responses. The server yields the same lines it logs, one per event. The client reads with `httpx.stream` and passes each line straight to `logger.info` — no JSON, no event parsing, client is just a pipe.

HTTP status code is committed in headers before streaming starts, so mid-stream errors are signalled as `ERROR: ...` lines instead of a 500 status.

---

## Gather saves on arrival (`backend/api.py`)

Previously gathered all shards into memory then saved at the end — a mid-gather failure discarded everything. Now `_gather_and_save` wraps pull + save together; each shard hits disk as soon as it arrives. Retry thread uses the same function.

---

## O(n²) receive bug (`networking/send_receive.py`)

**Problem:** The receive loop built the message buffer with `data += chunk`. In Python, `bytes` is immutable — every `+=` allocates a new object, copies all bytes received so far into it, then frees the old one. For a 169 MB shard received in 65 KB chunks (~2700 iterations), the total bytes copied is roughly `sum(1..2700) × 65 KB ≈ 240 GB`. On a Pi with slower memory, this triggered heap pressure and Linux swap activity on the SD card — turning a 2-minute transfer into 13 minutes.

**Solution:** Pre-allocate a single `bytearray` of the exact message length, wrap it in a `memoryview`, and use `recv_into` to write each chunk directly into the buffer at the right offset. Zero copies, zero allocations after the initial one.

```python
# before
data = b""
while remaining > 0:
    chunk = sock.recv(min(65536, remaining))
    data += chunk  # copies everything every iteration

# after
buf = bytearray(msglen)
view = memoryview(buf)
received = 0
while received < msglen:
    n = sock.recv_into(view[received:], min(65536, msglen - received))
    received += n
```

Result: pi4-4 transfer time dropped from 13 min to 2 min on a 676 MB checkpoint.

---

## macOS TCC + LaunchDaemon (`scripts/launch.sh`, `scripts/smoltorrent_startup.sh`)

macOS Transparency Consent and Control (TCC) blocks system daemons (running as root) from accessing `~/Desktop`, `~/Documents`, and `~/Downloads`. The project was originally at `~/Desktop/smoltorrent/` and checkpoints at `~/Desktop/smolcluster/checkpoints/` — the LaunchDaemon silently failed to reach these paths.

Fix: moved code to `~/smoltorrent/` and checkpoints to `~/smolcluster/checkpoints/`. The startup script is copied to `/usr/local/bin/` (TCC-safe) before being registered as a LaunchDaemon.

---

## macOS 26 Tahoe LaunchDaemon registration

Tahoe broke the traditional auto-start APIs:

- `launchctl load` → SIGABRT exit 134 (API removed)
- `launchctl bootstrap gui/<UID>` → error 125 (GUI domain broken in beta)
- `~/Library/LaunchAgents/` → silently ignored (needs SMAppService from Swift)

Working sequence:
```bash
sudo launchctl bootout system/com.smoltorrent.startup 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/com.smoltorrent.startup.plist
sudo launchctl enable system/com.smoltorrent.startup
```

The plist requires `UserName` (so it runs as the user, not root) and `EnvironmentVariables` with `PATH` including `/opt/homebrew/bin` (so `uv`, `tmux`, and `brew` are found at boot time).

`launchctl print system/com.smoltorrent.startup` showing `state = not running` and `last exit code = 1` after boot is expected — the daemon runs once, launches everything into tmux, then exits. The cluster runs independently in tmux.
