"""Worker orchestration: shard transfer, retry logic, and heartbeat checks."""

import logging
import mmap
import struct
import threading
import time
from queue import Queue

from networking.send_receive import send_message, receive_message
from utils.common_utils import connect_with_retry
from utils.check_workers import ping_worker

logger = logging.getLogger(__name__)

MAX_RETRIES = 6


def send_shard_to_worker(
    worker: dict,
    ckpt_path: str,
    file_offset: int,
    length: int,
    tensor_meta: dict,
    rel_path: str,
    shard_filename: str = "shard_0.safetensors",
    shard_index: int = -1,
    size_bytes: int = 0,
    source_path: str = "",
) -> tuple[bool, str, dict]:
    """Send one shard to a worker and record the placement in the tracker on success.

    The shard is never loaded into RAM. Flow:
      1. Stream-hash the tensor byte range in 1 MB chunks → checksum
      2. Send store_shard command + checksum to worker
      3. Send tensor metadata (mini safetensors header with rebased offsets)
      4. sendfile the tensor bytes directly from the original checkpoint file
      5. Wait for worker ack → on success, write to shard_tracker DB immediately

    Recording in the tracker here (not in the caller) means retries that succeed
    also get tracked, and the caller doesn't need to carry extra state just to
    write the DB entry later.

    Args:
        worker:        Worker config dict (ip, port, rank, ...).
        ckpt_path:     Absolute path to the original checkpoint on the master.
        file_offset:   Absolute byte offset in ckpt_path where this shard starts.
        length:        Number of tensor data bytes for this shard.
        tensor_meta:   {tensor_name: {dtype, shape, data_offsets}} rebased to 0.
        rel_path:      Relative checkpoint path — the tracker key.
        shard_filename: Filename saved on the worker (e.g. shard_0.safetensors).
        shard_index:   Which chunk of model data this covers (0, 1, ..., n-1).
        size_bytes:    Total checkpoint size in bytes (informational, stored in tracker).
        source_path:   Absolute path to original checkpoint (informational).

    Returns:
        (ok, error_msg, result) — result contains ``shard_path`` on success.
    """
    from utils.common_utils import compute_checksum
    from networking.send_receive import serve_file
    from utils.shard_tracker import add_shard

    rank = worker["rank"]
    host = worker.get("host") or worker.get("device", "")
    sock = None
    try:
        logger.info(
            "[ops] send_shard rank=%d shard=%s shard_index=%d offset=%d len=%d (%.1f MB)",
            rank, shard_filename, shard_index, file_offset, length, length / 1024**2,
        )
        # Pass 1: stream-hash the tensor byte range — OS page cache makes
        # the subsequent sendfile (pass 2) read from cache, not disk again.
        checksum = compute_checksum(ckpt_path, offset=file_offset, length=length)
        logger.debug("[ops] checksum for rank=%d shard=%s: %s…", rank, shard_filename, checksum[:16])

        sock = connect_with_retry(worker["ip"], worker["port"], rank)
        send_message(sock, ("store_shard", rank, checksum, rel_path, shard_filename))
        # Mini header lets the worker reconstruct a valid safetensors file
        send_message(sock, tensor_meta)
        # Pass 2: zero-copy stream of just this shard's tensor bytes
        serve_file(sock, ckpt_path, file_offset, length)

        response = receive_message(sock)
        sock.close()

        if response is None:
            return False, "no response from worker", {}
        if response[0] == "store_shard_done":
            _, _, shard_path = response
            logger.info("[ops] rank %d ack %s → %s", rank, shard_filename, shard_path)
            add_shard(
                rank=rank,
                shard_key=rel_path,
                host=host,
                shard_index=shard_index,
                shard_files=[shard_filename],
                size_bytes=size_bytes,
                source_path=source_path,
                checksum=checksum,
            )
            return True, "", {"shard_path": shard_path}
        _, _, err_msg = response
        logger.error("[ops] rank %d store failed: %s", rank, err_msg)
        return False, err_msg, {}
    except Exception as e:
        logger.exception("[ops] Unhandled error sending shard to rank %d", rank)
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        return False, str(e), {}



def gather_shard_data_only(
    worker: dict,
    rel_path: str,
    merged_mm: mmap.mmap,
    write_offset: int,
    data_length: int,
    shard_filename: str = "shard_0.safetensors",
) -> tuple[bool, str]:
    """Pull raw tensor bytes from a worker into an offset of the pre-allocated merged file.

    Uses the ``send_shard_range`` protocol: the worker strips its local safetensors
    header and streams only the tensor data bytes.  The coordinator writes them
    directly into *merged_mm* at *write_offset* — no Python tensor memory allocation.

    Thread-safe when called concurrently with non-overlapping (write_offset, data_length).

    Args:
        worker:        Worker config dict (ip, port, rank, ...).
        rel_path:      Checkpoint relative path (storage key on the worker).
        merged_mm:     Open read/write mmap of the pre-allocated merged safetensors file.
        write_offset:  Byte offset in *merged_mm* to start writing.
        data_length:   Expected number of tensor bytes for this shard.
        shard_filename: Filename on the worker (e.g. ``shard_0.safetensors``).

    Returns:
        ``(ok, error_msg)``
    """
    from networking.send_receive import receive_file

    rank = worker["rank"]
    sock = None
    try:
        logger.info(
            "[ops] gather_data rank=%d rel=%s write_offset=%d len=%d (%.1f MB)",
            rank, rel_path, write_offset, data_length, data_length / 1024**2,
        )
        sock = connect_with_retry(worker["ip"], worker["port"], rank)
        send_message(sock, ("send_shard_range", rank, rel_path, shard_filename))
        status = receive_message(sock)
        if status is None or status[0] != "send_shard_range_ok":
            sock.close()
            return False, f"send_shard_range rejected by rank {rank}: {status}"
        _, _, announced_length = status
        if announced_length != data_length:
            sock.close()
            return False, f"rank {rank} announced {announced_length} bytes but expected {data_length}"
        receive_file(sock, merged_mm, write_offset=write_offset, expected_length=data_length)
        sock.close()
        logger.info("[ops] gather_shard_data_only rank=%d offset=%d len=%d OK", rank, write_offset, data_length)
        return True, ""
    except Exception as e:
        logger.exception("[ops] Unhandled error in gather_shard_data_only rank=%d", rank)
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        return False, str(e)


def run_retry_worker(
    retry_queue: Queue, recovered: list, dead_letter: list, lock: threading.Lock
) -> None:
    """Daemon thread that drains *retry_queue* with exponential backoff.

    Each queue item is a dict with keys:
      * ``fn``      — zero-arg callable returning ``(ok, err, result)``.
      * ``worker``  — worker config dict (rank, host, …).
      * ``attempt`` — current attempt number (incremented on re-queue).

    Items that exceed ``MAX_RETRIES`` are moved to *dead_letter*; items that
    succeed are appended to *recovered*.  Both lists are guarded by *lock*.
    Runs until the process exits (never returns).

    Args:
        retry_queue:  :class:`queue.Queue` fed by the main store/gather loop.
        recovered:    Shared list; successful retries are appended here.
        dead_letter:  Shared list; permanently failed jobs are appended here.
        lock:         Threading lock protecting both shared lists.

    Returns:
        None.
    """
    while True:
        item = retry_queue.get()
        fn, worker, attempt = item["fn"], item["worker"], item["attempt"]
        rank = worker["rank"]
        if attempt > MAX_RETRIES:
            logger.error("[ops] Worker %d permanently failed after %d retries", rank, MAX_RETRIES)
            with lock:
                dead_letter.append({"rank": rank, "host": worker.get("host"), "error": "max retries exceeded"})
            retry_queue.task_done()
            continue
        delay = 2 ** attempt
        logger.info("[ops] Worker %d retry %d/%d — sleeping %ds", rank, attempt, MAX_RETRIES, delay)
        time.sleep(delay)
        ok, err, result = fn()
        if ok:
            logger.info("[ops] Worker %d recovered on attempt %d", rank, attempt)
            with lock:
                recovered.append({
                    "rank": worker["rank"],
                    "host": worker.get("host") or worker.get("device", ""),
                    **result,
                })
        else:
            logger.warning("[ops] Worker %d retry attempt %d failed: %s", rank, attempt, err)
            retry_queue.put({"fn": fn, "worker": worker, "attempt": attempt + 1})
        retry_queue.task_done()


def heartbeat_workers(workers: list[dict], timeout: float = 3.0) -> list[dict]:
    """Ping every configured worker and return the list of unreachable ones.

    Args:
        workers: List of worker config dicts from ``config.yaml``, each containing
                 at minimum ``rank``, ``ip``, and ``port``.
        timeout: Per-worker connect/receive timeout in seconds (default 3.0).

    Returns:
        List of dicts for workers that did not respond, each with keys
        ``rank`` and ``host``.  Empty list if all workers are alive.
    """
    dead = []
    for w in workers:
        rank = w["rank"]
        host = w.get("host") or w.get("device") or w["ip"]
        alive, reason = ping_worker(host, w["ip"], w["port"], rank, timeout=timeout)
        if not alive:
            logger.warning("[ops] Heartbeat failed for rank %d (%s): %s", rank, host, reason)
            dead.append({"rank": rank, "host": host})
    return dead
