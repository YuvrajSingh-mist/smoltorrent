"""Worker orchestration: shard transfer, retry logic, and heartbeat checks."""

import logging
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
) -> tuple[bool, str, dict]:
    """Send one shard to a worker with zero memory allocation on the master.

    The shard is never loaded into RAM. Flow:
      1. Stream-hash the tensor byte range in 1 MB chunks → checksum
      2. Send store_shard command + checksum to worker
      3. Send tensor metadata (mini safetensors header with rebased offsets)
      4. sendfile the tensor bytes directly from the original checkpoint file

    The worker calls receive_shard_mmap which stitches the header + bytes into
    a valid .safetensors file it can load with load_tensors().

    Args:
        worker: Worker config dict (ip, port, rank, ...).
        ckpt_path: Absolute path to the original checkpoint on the master.
        file_offset: Absolute byte offset in ckpt_path where this shard starts.
        length: Number of tensor data bytes for this shard.
        tensor_meta: {tensor_name: {dtype, shape, data_offsets}} with offsets
                     rebased to 0 (as returned by get_shard_ranges).
        rel_path: Relative checkpoint path used as the storage key on the worker.
        shard_filename: Filename to save as on the worker (e.g. shard_0.safetensors).

    Returns:
        (ok, error_msg, result) — result contains ``shard_path`` on success.
    """
    from utils.common_utils import compute_checksum
    from networking.send_receive import serve_file_range

    rank = worker["rank"]
    sock = None
    try:
        # Pass 1: stream-hash the tensor byte range — OS page cache makes
        # the subsequent sendfile (pass 2) read from cache, not disk again.
        checksum = compute_checksum(ckpt_path, offset=file_offset, length=length)

        sock = connect_with_retry(worker["ip"], worker["port"], rank)
        send_message(sock, ("store_shard", rank, checksum, rel_path, shard_filename))
        # Mini header lets the worker reconstruct a valid safetensors file
        send_message(sock, tensor_meta)
        # Pass 2: zero-copy stream of just this shard's tensor bytes
        serve_file_range(sock, ckpt_path, file_offset, length)

        response = receive_message(sock)
        sock.close()
        
        if response is None:
            return False, "no response from worker", {}
        if response[0] == "store_shard_done":
            _, _, shard_path = response
            logger.info("[ops] Worker %d acknowledged %s → %s", rank, shard_filename, shard_path)
            return True, "", {"shard_path": shard_path}
        _, _, err_msg = response
        logger.error("[ops] Worker %d store failed: %s", rank, err_msg)
        return False, err_msg, {}
    except Exception as e:
        logger.exception("[ops] Unhandled error sending shard to rank %d", rank)
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        return False, str(e), {}


def gather_shard_from_worker(
    worker: dict, rel_path: str, dest_path: str,
    shard_filename: str = "shard_0.safetensors",
) -> tuple[bool, str]:
    """Pull one shard from a worker directly into dest_path via mmap.

    Returns:
        (ok, error_msg)
    """
    rank = worker["rank"]
    sock = None
    try:
        from networking.send_receive import receive_file_mmap
        sock = connect_with_retry(worker["ip"], worker["port"], rank)
        send_message(sock, ("send_shard", rank, rel_path, shard_filename))
        status = receive_message(sock)
        if status is None or status[0] != "send_shard_ok":
            sock.close()
            return False, f"shard not available on rank {rank}: {status}"
        receive_file_mmap(sock, dest_path)
        sock.close()
        return True, ""
    except Exception as e:
        logger.exception("[ops] Unhandled error gathering shard from rank %d", rank)
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        return False, str(e)


def run_retry_worker(
    retry_queue: Queue, recovered: list, dead_letter: list, lock: threading.Lock
) -> None:
    """Daemon thread that drains retry_queue with exponential backoff.

    Each item must have keys: ``fn`` (zero-arg callable → (ok, err, result)),
    ``worker``, ``attempt``.
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
        time.sleep(2 ** attempt)
        ok, err, result = fn()
        if ok:
            with lock:
                recovered.append(result)
        else:
            logger.warning("[ops] Worker %d retry attempt %d failed: %s", rank, attempt, err)
            retry_queue.put({"fn": fn, "worker": worker, "attempt": attempt + 1})
        retry_queue.task_done()


def heartbeat_workers(workers: list[dict], timeout: float = 3.0) -> list[dict]:
    """Ping every worker. Returns list of unreachable workers."""
    dead = []
    for w in workers:
        rank = w["rank"]
        host = w.get("host") or w.get("device") or w["ip"]
        alive, reason = ping_worker(host, w["ip"], w["port"], rank, timeout=timeout)
        if not alive:
            logger.warning("[ops] Heartbeat failed for rank %d (%s): %s", rank, host, reason)
            dead.append({"rank": rank, "host": host})
    return dead
