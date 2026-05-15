"""FastAPI server that orchestrates shard distribution across workers.

Exposes two endpoints:
  POST /store-shard  — shards a checkpoint and pushes each shard to its ranked worker.
  POST /gather-shards — pulls shards from all workers, saves locally, then merges.
"""
import logging
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import uvicorn
import yaml
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).parents[1]))

from networking.send_receive import receive_message, send_message
from utils.common_utils import chunk_data, compute_checksum, load_tensors, merge_shards, save_merged_model, shard_from_bytes, shard_to_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SmolTorrent Shard API")

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"
SHARDS_ROOT = Path(__file__).parents[1] / "shards"
MAX_RETRIES = 3


def _load_config() -> dict:
    """Load and return the YAML config from the default config path.

    Returns:
        Parsed config dict.
    """
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _send_shard_to_worker(worker: dict, shard_bytes: bytes, checksum: str, rel_path: str) -> tuple[bool, str, dict]:
    """Send one pre-serialized shard to a worker and verify the ack.

    Args:
        worker: Worker config dict with keys ``rank``, ``ip``, ``port``, and optionally ``host``/``device``.
        shard_bytes: Already-serialized safetensors bytes.
        checksum: SHA-256 hex digest of ``shard_bytes``.
        rel_path: Relative path from ckpt_root (e.g. ``grpo/run1/step_100``). Worker stores shard here.

    Returns:
        Tuple of (ok, error_msg, result) where result contains ``shard_path`` on success.
    """
    rank = worker["rank"]
    try:
        sock = _connect_with_retry(worker["ip"], worker["port"], rank)
        send_message(sock, ("store_shard", rank, shard_bytes, checksum, rel_path))
        response = receive_message(sock)
        sock.close()
        if response is None:
            return False, "no response from worker", {}
        if response[0] == "store_shard_done":
            _, _, shard_path = response
            logger.info("Worker %d acknowledged shard storage → %s", rank, shard_path)
            return True, "", {"shard_path": shard_path}
        _, _, err_msg = response
        logger.error("Worker %d store failed: %s", rank, err_msg)
        return False, err_msg, {}
    except Exception as e:
        logger.exception("Unhandled error sending shard to rank %d", rank)
        return False, str(e), {}


def _gather_shard_from_worker(worker: dict, rel_path: str) -> tuple[bool, str, dict]:
    """Pull one shard from a worker.

    Args:
        worker: Worker config dict.
        rel_path: Relative path identifying which checkpoint shard to pull.

    Returns:
        Tuple of (ok, error_msg, result) where result contains ``rank``, ``host``, and ``_shard``.
    """
    rank = worker["rank"]
    host = worker.get("host") or worker.get("device")
    try:
        sock = _connect_with_retry(worker["ip"], worker["port"], rank)
        send_message(sock, ("send_shard", rank, rel_path))
        shard_bytes = receive_message(sock)
        sock.close()
        if shard_bytes is None:
            return False, "no shard received", {}
        return True, "", {"rank": rank, "host": host, "_shard": shard_from_bytes(shard_bytes)}
    except Exception as e:
        logger.exception("Unhandled error gathering shard from rank %d", rank)
        return False, str(e), {}


def _retry_worker(retry_queue: Queue, recovered: list, dead_letter: list, lock: threading.Lock, send_fn) -> None:
    """Daemon thread that drains retry_queue with exponential backoff.

    Args:
        retry_queue: Queue of dicts with keys ``worker``, ``shard``, ``checksum``, ``attempt``.
        recovered: Shared list; successful results are appended here under ``lock``.
        dead_letter: Shared list; permanently failed entries are appended here under ``lock``.
        lock: Threading lock protecting ``recovered`` and ``dead_letter``.
        send_fn: Callable ``(worker, shard, checksum) -> (ok, err, result)`` — rel_path already bound via closure.
    """
    while True:
        item = retry_queue.get()
        worker, shard, checksum, attempt = item["worker"], item["shard"], item["checksum"], item["attempt"]
        rank = worker["rank"]
        if attempt > MAX_RETRIES:
            logger.error("Worker %d permanently failed after %d retries", rank, MAX_RETRIES)
            with lock:
                dead_letter.append({"rank": rank, "host": worker.get("host"), "error": "max retries exceeded"})
            retry_queue.task_done()
            continue
        time.sleep(2 ** attempt)
        ok, err, result = send_fn(worker, shard, checksum)
        if ok:
            with lock:
                recovered.append(result)
        else:
            logger.warning("Worker %d retry attempt %d failed: %s", rank, attempt, err)
            retry_queue.put({"worker": worker, "shard": shard, "checksum": checksum, "attempt": attempt + 1})
        retry_queue.task_done()


def _connect_with_retry(ip: str, port: int, rank: int, retries: int = 3, delay: float = 2.0) -> socket.socket:
    """Open a TCP connection, retrying on failure with a fixed delay.

    Args:
        ip: Target IP address.
        port: Target port.
        rank: Worker rank used only for log messages.
        retries: Maximum number of connection attempts.
        delay: Seconds to wait between attempts.

    Returns:
        Connected blocking socket.

    Raises:
        ConnectionError: If all attempts fail.
    """
    for attempt in range(1, retries + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            logger.info(f"Connecting to rank {rank} at {ip}:{port} (attempt {attempt}/{retries})")
            sock.connect((ip, port))
            sock.settimeout(None)
            logger.info(f"Connected to rank {rank} at {ip}:{port}")
            return sock
        except (OSError, ConnectionRefusedError) as e:
            sock.close()
            logger.warning(f"Attempt {attempt}/{retries} failed for rank {rank} at {ip}:{port}: {e}")
            if attempt < retries:
                time.sleep(delay)
    raise ConnectionError(f"Could not connect to rank {rank} at {ip}:{port} after {retries} attempts")


@app.post("/store-shard")
def store_shard(ckpt_path: str = Query(..., description="Absolute path to the checkpoint .safetensors file on master")):
    """Load a checkpoint, shard it, push each shard to its ranked worker, stream log lines.

    Args:
        ckpt_path: Absolute path to the checkpoint file. Relative path from ``ckpt_root``
                   in config is computed automatically and used as the storage path on workers.

    Returns:
        StreamingResponse (text/plain) — one log line per event.
    """
    def _generate():
        def _log(msg: str) -> str:
            logger.info(msg)
            return msg + "\n"

        config = _load_config()
        workers = config["devices_config"]["workers"]
        num_workers = len(workers)
        ckpt_root = Path(config["ckpt_root"]).expanduser()
        ckpt_file = Path(ckpt_path).expanduser()

        if not ckpt_file.exists():
            yield f"ERROR: checkpoint not found: {ckpt_file}\n"
            return

        try:
            rel_path = str(ckpt_file.parent.relative_to(ckpt_root))
        except ValueError:
            yield f"ERROR: {ckpt_file} is not under ckpt_root {ckpt_root}\n"
            return

        tensors = load_tensors(ckpt_file)
        total_mb = sum(v.nbytes for v in tensors.values()) / 1024**2
        yield _log(f"Loaded {len(tensors)} tensors ({total_mb:.1f} MB) from {rel_path} — chunking into {num_workers} shards")
        chunks = chunk_data(tensors, n_chunks=num_workers)

        sent: list = []
        dead_letter: list = []
        lock = threading.Lock()
        store_queue: Queue = Queue()

        def _send(worker, shard_bytes, checksum):
            return _send_shard_to_worker(worker, shard_bytes, checksum, rel_path)

        threading.Thread(target=_retry_worker, args=(store_queue, sent, dead_letter, lock, _send), daemon=True).start()

        # Serialize all shards up front, then fire all sends in parallel
        jobs = []
        for i, worker in enumerate(workers):
            shard_bytes = shard_to_bytes(chunks[i])
            checksum = compute_checksum(shard_bytes)
            jobs.append((worker, shard_bytes, checksum))

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            future_to_job = {
                pool.submit(_send, worker, shard_bytes, checksum): (worker, shard_bytes, checksum)
                for worker, shard_bytes, checksum in jobs
            }
            for future in as_completed(future_to_job):
                worker, shard_bytes, checksum = future_to_job[future]
                rank = worker["rank"]
                host = worker.get("host") or worker.get("device")
                ok, err, result = future.result()
                if ok:
                    with lock:
                        sent.append({"rank": rank, "host": host, "shard_path": result.get("shard_path")})
                    yield _log(f"  ✓ rank {rank} ({host})")
                else:
                    yield _log(f"  ↻ rank {rank} ({host}) failed — queuing retry: {err}")
                    store_queue.put({"worker": worker, "shard": shard_bytes, "checksum": checksum, "attempt": 1})

        store_queue.join()

        with lock:
            failed = list(dead_letter)
            succeeded = list(sent)

        for f in failed:
            yield _log(f"  ✗ rank {f['rank']} ({f.get('host')}) permanently failed: {f.get('error')}")

        if failed:
            yield f"ERROR: {len(failed)}/{num_workers} shards failed\n"
        else:
            yield _log(f"Done: {len(succeeded)}/{num_workers} shards stored → {rel_path}")

    return StreamingResponse(_generate(), media_type="text/plain")


@app.post("/gather-shards")
def gather_shards(ckpt_path: str = Query(..., description="Absolute path to the checkpoint file (same path used for store)")):
    """Pull shards from every worker, save each as it arrives, merge, stream log lines.

    Args:
        ckpt_path: Absolute path to the checkpoint file. Relative path from ``ckpt_root``
                   is computed automatically — must be the same file used during store.

    Returns:
        StreamingResponse (text/plain) — one log line per event.
    """
    def _generate():
        def _log(msg: str) -> str:
            logger.info(msg)
            return msg + "\n"

        config = _load_config()
        workers = config["devices_config"]["workers"]
        ckpt_root = Path(config["ckpt_root"]).expanduser()
        ckpt_file = Path(ckpt_path).expanduser()

        try:
            rel_path = str(ckpt_file.parent.relative_to(ckpt_root))
        except ValueError:
            yield f"ERROR: {ckpt_file} is not under ckpt_root {ckpt_root}\n"
            return

        gathered: list = []
        shards_by_rank: dict = {}
        save_errors: list = []
        dead_letter: list = []
        lock = threading.Lock()
        gather_queue: Queue = Queue()

        def _gather_and_save(worker: dict, _shard, _checksum) -> tuple[bool, str, dict]:
            """Pull shard for ``rel_path`` from ``worker`` and save to local SHARDS_ROOT.

            Args:
                worker: Worker config dict.
                _shard: Unused; for retry-worker signature compatibility.
                _checksum: Unused; for retry-worker signature compatibility.

            Returns:
                Tuple of (ok, error_msg, result_entry).
            """
            ok, err, result = _gather_shard_from_worker(worker, rel_path)
            if not ok:
                return False, err, {}
            rank = result["rank"]
            host = result["host"]
            received_shard = result.pop("_shard")
            shard_dir = SHARDS_ROOT / f"worker_{rank}" / rel_path
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / "shard.safetensors"
            try:
                from safetensors.torch import save_file
                save_file(received_shard, str(shard_path))
            except Exception as e:
                with lock:
                    save_errors.append({"rank": rank, "host": host, "error": str(e)})
                return False, str(e), {}
            result["shard_path"] = str(shard_path)
            with lock:
                shards_by_rank[rank] = received_shard
            return True, "", result

        threading.Thread(target=_retry_worker, args=(gather_queue, gathered, dead_letter, lock, _gather_and_save), daemon=True).start()

        for worker in workers:
            rank = worker["rank"]
            host = worker.get("host") or worker.get("device")
            ok, err, result = _gather_and_save(worker, {}, "")
            if ok:
                with lock:
                    gathered.append(result)
                yield _log(f"  ✓ rank {rank} ({host}) — saved → {result['shard_path']}")
            else:
                yield _log(f"  ↻ rank {rank} ({host}) failed — queuing retry: {err}")
                gather_queue.put({"worker": worker, "shard": {}, "checksum": "", "attempt": 1})

        gather_queue.join()

        with lock:
            all_gathered = list(gathered)
            failed = list(dead_letter) + list(save_errors)

        if failed:
            for f in failed:
                yield _log(f"  ✗ rank {f['rank']} ({f.get('host')}): {f['error']}")
            yield f"ERROR: {len(failed)}/{len(workers)} shards failed — skipping merge\n"
            return

        save_path = Path(config["ckpt_root"]).expanduser() / rel_path / "merged.safetensors"
        yield _log(f"Merging {len(all_gathered)} shards → {save_path}")
        merged = merge_shards(list(shards_by_rank.values()))
        save_merged_model(merged, save_path)
        yield _log(f"Done: saved → {save_path}")

    return StreamingResponse(_generate(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
