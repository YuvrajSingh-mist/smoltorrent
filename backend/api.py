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
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

sys.path.insert(0, str(Path(__file__).parents[1]))

from networking.send_receive import receive_message, send_message, network_metrics
from utils.common_utils import (
    chunk_data,
    compute_checksum,
    load_tensors,
    merge_shards,
    save_merged_model,
    shard_from_bytes,
    shard_to_bytes,
    save_shard,
)
from utils.network_metrics import log_metrics
from discovery import discover_workers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SmolTorrent Shard API")
app.mount("/metrics", make_asgi_app())  # Prometheus scrape endpoint

_store_ops = Counter("smoltorrent_store_operations_total", "Completed store operations")
_gather_ops = Counter(
    "smoltorrent_gather_operations_total", "Completed gather operations"
)
_xfer_errors = Counter(
    "smoltorrent_transfer_errors_total", "Transfer errors by worker rank", ["rank"]
)

_WALL_BUCKETS = [10, 30, 60, 120, 180, 240, 300, 420, 600]
_store_wall = Histogram(
    "smoltorrent_store_wall_seconds",
    "End-to-end wall-clock time of /store-shard",
    buckets=_WALL_BUCKETS,
)
_gather_wall = Histogram(
    "smoltorrent_gather_wall_seconds",
    "End-to-end wall-clock time of /gather-shards",
    buckets=_WALL_BUCKETS,
)

# process_start_time_seconds is not emitted by prometheus_client on macOS
# (ProcessCollector uses /proc which doesn't exist). Expose it manually.
_process_start_time = Gauge(
    "process_start_time_seconds", "Unix timestamp when this process started"
)
_process_start_time.set(time.time())

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"


# Pre-initialise per-rank error series so they appear in Prometheus at zero
# before any failure occurs (labeled counters are only emitted after first inc()).
def _init_error_labels() -> None:
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    for w in cfg["devices_config"]["workers"]:
        _xfer_errors.labels(rank=str(w["rank"]))


try:
    _init_error_labels()
except Exception:
    pass  # config missing at import time — labels will register on first use
SHARDS_ROOT = Path(__file__).parents[1] / "shards"
MAX_RETRIES = 6
REDUNDANCY = 2  # replicas per shard (1 = no redundancy, 2 = one primary + one replica)


def _load_config() -> dict:
    """Load and return the YAML config from the default config path.

    Returns:
        Parsed config dict.
    """
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _send_shard_to_worker(
    worker: dict, shard_bytes: bytes, checksum: str, rel_path: str
) -> tuple[bool, str, dict]:
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
            logger.info("[api] Worker %d acknowledged shard storage → %s", rank, shard_path)
            return True, "", {"shard_path": shard_path}
        _, _, err_msg = response
        logger.error("[api] Worker %d store failed: %s", rank, err_msg)
        return False, err_msg, {}
    except Exception as e:
        logger.exception("[api] Unhandled error sending shard to rank %d", rank)
        try:
            sock.close()
        except Exception:
            pass
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
        return (
            True,
            "",
            {"rank": rank, "host": host, "_shard": shard_from_bytes(shard_bytes)},
        )
    except Exception as e:
        logger.exception("[api] Unhandled error gathering shard from rank %d", rank)
        return False, str(e), {}


def _retry_worker(
    retry_queue: Queue, recovered: list, dead_letter: list, lock: threading.Lock
) -> None:
    """Daemon thread that drains retry_queue with exponential backoff.

    Each item must have keys: ``fn`` (zero-arg callable -> (ok, err, result)),
    ``worker`` (for logging/dead-letter), ``attempt``.
    """
    while True:
        item = retry_queue.get()
        fn, worker, attempt = item["fn"], item["worker"], item["attempt"]
        rank = worker["rank"]
        if attempt > MAX_RETRIES:
            logger.error(
                "[api] Worker %d permanently failed after %d retries", rank, MAX_RETRIES
            )
            with lock:
                dead_letter.append(
                    {
                        "rank": rank,
                        "host": worker.get("host"),
                        "error": "max retries exceeded",
                    }
                )
            retry_queue.task_done()
            continue
        time.sleep(2**attempt)
        ok, err, result = fn()
        if ok:
            with lock:
                recovered.append(result)
        else:
            logger.warning("[api] Worker %d retry attempt %d failed: %s", rank, attempt, err)
            retry_queue.put({"fn": fn, "worker": worker, "attempt": attempt + 1})

        retry_queue.task_done()


def _connect_with_retry(
    ip: str, port: int, rank: int, retries: int = 3, delay: float = 2.0
) -> socket.socket:
    """Open a TCP connection, retrying on failure with a fixed delay.

    Args:
        ip: Target IP address.
        port: Target port.
        rank: Worker rank used only for log messages.
        retries: Maximum number of connection attempts.
        delay: Base delay in seconds — actual wait is delay * 2^(attempt-1).

    Returns:
        Connected blocking socket.

    Raises:
        ConnectionError: If all attempts fail.
    """
    for attempt in range(1, retries + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            logger.info(
                f"[api] Connecting to rank {rank} at {ip}:{port} (attempt {attempt}/{retries})"
            )
            sock.connect((ip, port))
            sock.settimeout(
                None
            )  # blocking for send/receive — large shards need no deadline
            logger.info(f"[api] Connected to rank {rank} at {ip}:{port}")
            return sock
        except (OSError, ConnectionRefusedError) as e:
            sock.close()
            logger.warning(
                f"[api] Attempt {attempt}/{retries} failed for rank {rank} at {ip}:{port}: {e}"
            )
            if attempt < retries:
                time.sleep(delay * (2 ** (attempt - 1)))
    raise ConnectionError(
        f"Could not connect to rank {rank} at {ip}:{port} after {retries} attempts"
    )


@app.post("/store-shard")
def store_shard(
    ckpt_path: str = Query(
        ..., description="Absolute path to the checkpoint .safetensors file on master"
    ),
):
    """Load a checkpoint, shard it, push each shard to its ranked worker, stream log lines.

    Args:
        ckpt_path: Absolute path to the checkpoint file. Relative path from ``ckpt_root``
                   in config is computed automatically and used as the storage path on workers.

    Returns:
        StreamingResponse (text/plain) — one log line per event.
    """

    def _generate():
        def _log(msg: str) -> str:
            logger.info("[api] %s", msg)
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

        _store_start = time.monotonic()
        tensors = load_tensors(ckpt_file)
        total_mb = sum(v.nbytes for v in tensors.values()) / 1024**2
        yield _log(
            f"Loaded {len(tensors)} tensors ({total_mb:.1f} MB) from {rel_path} — chunking into {num_workers} shards"
        )
        chunks = chunk_data(tensors, n_chunks=num_workers)

        sent: list = []
        dead_letter: list = []
        lock = threading.Lock()
        store_queue: Queue = Queue()

        threading.Thread(
            target=_retry_worker,
            args=(store_queue, sent, dead_letter, lock),
            daemon=True,
        ).start()

        # Serialize all shards up front (decoupled from worker assignment)
        shards = []
        for i in range(num_workers):
            sb = shard_to_bytes(chunks[i])
            shards.append((sb, compute_checksum(sb)))

        # Round 0: shard i → workers[i]  (primary)
        # Round 1: shard i → workers[(i+1) % n]  (replica)
        # Rounds are sent sequentially so each Pi never handles two concurrent
        # 235 MB receives at the same time (which causes BrokenPipe under memory pressure).
        for round_idx in range(REDUNDANCY):
            round_jobs = [
                (workers[(i + round_idx) % num_workers], sb, cs)
                for i, (sb, cs) in enumerate(shards)
            ]
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                future_to_job = {
                    pool.submit(
                        _send_shard_to_worker, worker, shard_bytes, checksum, rel_path
                    ): (worker, shard_bytes, checksum)
                    for worker, shard_bytes, checksum in round_jobs
                }
                for future in as_completed(future_to_job):
                    worker, shard_bytes, checksum = future_to_job[future]
                    rank = worker["rank"]
                    host = worker.get("host") or worker.get("device")
                    ok, err, result = future.result()
                    if ok:
                        with lock:
                            sent.append(
                                {
                                    "rank": rank,
                                    "host": host,
                                    "shard_path": result.get("shard_path"),
                                }
                            )
                        yield _log(f"  ✓ rank {rank} ({host}) [round {round_idx}]")
                    else:
                        yield _log(
                            f"  ↻ rank {rank} ({host}) failed — queuing retry: {err}"
                        )
                        store_queue.put(
                            {
                                "fn": lambda w=worker, sb=shard_bytes, cs=checksum: (
                                    _send_shard_to_worker(w, sb, cs, rel_path)
                                ),
                                "worker": worker,
                                "attempt": 1,
                            }
                        )

        store_queue.join()

        failed = list(dead_letter)
        succeeded = list(sent)

        for f in failed:
            _xfer_errors.labels(rank=str(f["rank"])).inc()
            yield _log(
                f"  ✗ rank {f['rank']} ({f.get('host')}) permanently failed: {f.get('error')}"
            )

        total_expected = num_workers * REDUNDANCY
        log_metrics(network_metrics.get_metrics(), logger, "store")
        _store_wall.observe(time.monotonic() - _store_start)
        if failed:
            yield f"ERROR: {len(failed)}/{total_expected} sends failed\n"
        else:
            _store_ops.inc()
            yield _log(
                f"Done: {len(succeeded)}/{total_expected} sends ({REDUNDANCY}x replicated) → {rel_path}"
            )

    return StreamingResponse(_generate(), media_type="text/plain")


@app.post("/gather-shards")
def gather_shards(
    ckpt_path: str = Query(
        ...,
        description="Absolute path to the checkpoint file (same path used for store)",
    ),
):
    """Pull shards from every worker, save each as it arrives, merge, stream log lines.

    Args:
        ckpt_path: Absolute path to the checkpoint file. Relative path from ``ckpt_root``
                   is computed automatically — must be the same file used during store.

    Returns:
        StreamingResponse (text/plain) — one log line per event.
    """

    def _generate():
        def _log(msg: str) -> str:
            logger.info("[api] %s", msg)
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

        _gather_start = time.monotonic()

        gathered: list = []
        # Keyed by shard index (0..n-1), not worker rank. Matters when a replica
        # serves a shard: e.g. shard 0 falling back to workers[1] (rank 1) would
        # land in the wrong merge slot if keyed by rank. Equivalently , its just like
        # # primary: rank i → shards_by_rank[i]         ✓
        # replica: rank i+1 → shards_by_rank[i+1 - 1] ✓

        shards_by_index: dict = {}
        save_errors: list = []
        dead_letter: list = []
        lock = threading.Lock()
        gather_queue: Queue = Queue()
        num_workers = len(workers)

        def _gather_and_save(worker: dict, shard_index: int) -> tuple[bool, str, dict]:
            """Pull shard for ``rel_path`` from ``worker`` and save to local SHARDS_ROOT."""
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
                save_shard(received_shard, str(shard_path))
            except Exception as e:
                with lock:
                    save_errors.append({"rank": rank, "host": host, "error": str(e)})
                return False, str(e), {}
            result["shard_path"] = str(shard_path)
            with lock:
                shards_by_index[shard_index] = received_shard
            return True, "", result

        threading.Thread(
            target=_retry_worker,
            args=(gather_queue, gathered, dead_letter, lock),
            daemon=True,
        ).start()

        def _gather_one(i: int, worker: dict):
            rank = worker["rank"]
            host = worker.get("host") or worker.get("device")
            ok, err, result = _gather_and_save(worker, shard_index=i)
            if not ok and REDUNDANCY > 1:
                replica = workers[(i + 1) % num_workers]
                ok, err, result = _gather_and_save(replica, shard_index=i)
                if not ok:
                    return i, rank, host, False, err, result
                return (
                    i,
                    replica["rank"],
                    replica.get("host") or replica.get("device"),
                    True,
                    "",
                    result,
                )
            return i, rank, host, ok, err, result

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            future_to_idx = {
                pool.submit(_gather_one, i, w): i for i, w in enumerate(workers)
            }
            for future in as_completed(future_to_idx):
                i, rank, host, ok, err, result = future.result()
                if ok:
                    gathered.append(result)
                    yield _log(f"  ✓ shard {i} — saved → {result['shard_path']}")
                else:
                    yield _log(
                        f"  ↻ shard {i} (rank {rank}) failed — queuing retry: {err}"
                    )
                    worker = workers[i]
                    gather_queue.put(
                        {
                            "fn": lambda w=worker, si=i: _gather_and_save(w, si),
                            "worker": worker,
                            "attempt": 1,
                        }
                    )

        gather_queue.join()

        all_gathered = list(gathered)
        failed = list(dead_letter) + list(save_errors)

        if failed:
            for f in failed:
                _xfer_errors.labels(rank=str(f["rank"])).inc()
                yield _log(f"  ✗ rank {f['rank']} ({f.get('host')}): {f['error']}")
            yield f"ERROR: {len(failed)}/{len(workers)} shards failed — skipping merge\n"
            return

        save_path = (
            Path(config["ckpt_root"]).expanduser() / rel_path / "merged.safetensors"
        )
        yield _log(f"Merging {len(all_gathered)} shards → {save_path}")
        merged = merge_shards([shards_by_index[i] for i in range(num_workers)])
        save_merged_model(merged, save_path)
        log_metrics(network_metrics.get_metrics(), logger, "gather")
        _gather_wall.observe(time.monotonic() - _gather_start)
        _gather_ops.inc()
        yield _log(f"Done: saved → {save_path}")

    return StreamingResponse(_generate(), media_type="text/plain")


@app.get("/discover")
def discover(
    timeout: float = Query(10.0, description="How long to scan for workers (seconds)"),
):
    """Scan the local network for smoltorrent worker nodes.

    Uses mDNS (works on all platforms over WiFi/Ethernet) and AirDrop/AWDL
    on macOS. Workers must be running with discovery enabled (default when
    started via ``worker.py``).

    Returns:
        JSON list of found workers sorted by rank::

            {"workers": [{"ip": "...", "port": 5001, "rank": 1, "hostname": "pi4-1"}, ...]}
    """
    workers = discover_workers(timeout=timeout)
    logger.info("[api] Discovery found %d worker(s): %s", len(workers), workers)
    return {"workers": workers}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
