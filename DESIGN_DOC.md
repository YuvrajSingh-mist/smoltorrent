# SmolTorrent — Design Decisions

---

## Wire format — safetensors (`utils/common_utils.py`)

Mac runs MLX, Pi workers run torch. Pickle fails because unpickling an MLX array on Pi tries to `import mlx` which isn't installed. Numpy fails on bfloat16 because numpy doesn't natively support it — any path through numpy's dtype interpreter raises a PEP 3118 item-size mismatch.

Safetensors is the fix: it's a flat format (shape + dtype string + raw bytes) with no framework embedded. Same bytes are readable by both `mx.load()` and `safetensors.torch.load()`.

Serialization path on Mac (`shard_to_bytes`):
1. `bytes(mlx_array)` — MLX exposes raw memory via the buffer protocol directly, no numpy dtype interpretation
2. `torch.frombuffer(bytearray(...), dtype=...)` — reinterprets those bits as the correct torch dtype
3. `safetensors.torch.save(torch_dict)` — produces the bytes that go on the wire

On Pi: `safetensors.torch.load(received_bytes)` → torch tensors → saved to disk as `.safetensors`.

Safetensors is used as the serialization format for the wire, not just for disk storage — it's the only format that carries shape + dtype + tensor name cleanly across frameworks.

---

## Reliability — retry queue + checksum (`backend/api.py`)

A flaky worker should not block the other three or silently corrupt data.

- SHA-256 checksum computed on shard bytes before sending; worker verifies on receipt
- Failed sends go onto a daemon retry thread with exponential backoff (`2^attempt` seconds, max 3 retries)
- Main loop does not wait on failures — all workers are dispatched first, retry thread runs alongside
- `store_queue.join()` is the single wait point before returning — caller always gets a definitive result
- Permanently failed shards go to `dead_letter` and are reported in the response

---

## Double serialization bug (`backend/api.py`)

`/store-shard` serialized each shard once (to compute checksum), then passed the raw dict to `_send_shard_to_worker` which serialized it again after the socket was open. Second serialization stalled or raised mid-connection → worker saw "Socket connection broken".

Fix: `_send_shard_to_worker` takes `shard_bytes: bytes` directly. Serialized once, reused for both checksum and send.

---

## Socket timeout bug (`backend/api.py`)

`_connect_with_retry` set `sock.settimeout(2.0)` for the connect attempt and never cleared it. Every `sendall` on the returned socket had a 2s deadline. A 70 MB shard over Tailscale/WiFi blows past that.

Fix: `sock.settimeout(None)` immediately after `sock.connect()` succeeds. `send_message` also calls it before every send.

---

## Streaming progress (`backend/api.py`, `utils/shard_ops.py`)

Both endpoints return `text/plain` streaming responses. The server yields the same lines it logs, one per event. The client reads with `httpx.stream` and passes each line straight to `logger.info` — no JSON, no event parsing, client is just a pipe.

HTTP status code is committed in headers before streaming starts, so mid-stream errors are signalled as `ERROR: ...` lines instead of a 500 status.

---

## Gather saves on arrival (`backend/api.py`)

Previously gathered all shards into memory then saved at the end — a mid-gather failure discarded everything. Now `_gather_and_save` wraps pull + save together; each shard hits disk as soon as it arrives. Retry thread uses the same function.

---

## server.py removal

Legacy parameter-server that duplicated `/store-shard` logic with no checksums or retries. Removed from codebase and `launch.sh`. Workers now just start the shard listener and block.

---

## Multi-worker concurrency

Each incoming TCP connection on a worker spawns a daemon thread. Multiple clients (or the API retrying) can connect simultaneously without blocking each other.

---

## Heartbeat + shard pre-check (`main.py`, `utils/check_workers.py`)

Before gather: TCP ping every worker, then SSH to count `.safetensors` files on disk. Only proceeds if all `N` shards are present. Avoids triggering a gather that will partially fail.

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
