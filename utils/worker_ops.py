"""Worker orchestration: shard transfer, retry logic, and heartbeat checks."""

import logging
import threading
import time
from queue import Queue

from networking.send_receive import send_message, receive_message
from utils.common_utils import connect_with_retry
from utils.check_workers import ping_worker
from utils.common_utils import shard_from_bytes

logger = logging.getLogger(__name__)

MAX_RETRIES = 6


def send_shard_to_worker(
    worker: dict, shard_bytes: bytes, checksum: str, rel_path: str
) -> tuple[bool, str, dict]:
    """Send one serialized shard to a worker and verify the ack.

    Returns:
        (ok, error_msg, result) — result contains ``shard_path`` on success.
    """
    rank = worker["rank"]
    sock = None
    try:
        sock = connect_with_retry(worker["ip"], worker["port"], rank)
        send_message(sock, ("store_shard", rank, shard_bytes, checksum, rel_path))
        response = receive_message(sock)
        sock.close()
        if response is None:
            return False, "no response from worker", {}
        if response[0] == "store_shard_done":
            _, _, shard_path = response
            logger.info("[ops] Worker %d acknowledged shard → %s", rank, shard_path)
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


def gather_shard_from_worker(worker: dict, rel_path: str) -> tuple[bool, str, dict]:
    """Pull one shard from a worker.

    Returns:
        (ok, error_msg, result) — result contains ``rank``, ``host``, ``_shard``.
    """
    rank = worker["rank"]
    host = worker.get("host") or worker.get("device")
    try:
        sock = connect_with_retry(worker["ip"], worker["port"], rank)
        send_message(sock, ("send_shard", rank, rel_path))
        shard_bytes = receive_message(sock)
        sock.close()
        if shard_bytes is None:
            return False, "no shard received", {}
        return True, "", {"rank": rank, "host": host, "_shard": shard_from_bytes(shard_bytes)}
    except Exception as e:
        logger.exception("[ops] Unhandled error gathering shard from rank %d", rank)
        return False, str(e), {}


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
