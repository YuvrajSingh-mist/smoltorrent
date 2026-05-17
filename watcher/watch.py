"""Filesystem watcher — syncs new checkpoints to workers as they appear.

Flow per trigger:
  1. Sync all workers in parallel → union of rel_paths they already have.
  2. Scan ckpt_root locally for files matching extensions.
  3. Transfer only the diff (local - workers).
  4. After batch: re-evaluate pending (unstable-at-detection files).
     Stable ones re-trigger the loop; still-unstable stay in pending.
"""
import argparse
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.metrics import init_watcher_metrics, WatcherMetrics

_metrics: WatcherMetrics | None = None
from networking.send_receive import receive_message, send_message
from utils.shard_ops import request_store_shards

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smoltorrent.watcher")

_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"


def _load_config() -> dict:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _is_stable(path: Path, wait: float = 1.0) -> bool:
    """Return True if file size hasn't changed after wait seconds."""
    try:
        before = path.stat().st_size
        time.sleep(wait)
        return path.stat().st_size == before
    except OSError:
        return False


def _sync_worker(worker: dict, extensions: list[str]) -> tuple[bool, set[str]]:
    """Ask one worker what rel_paths it already has.
    Returns (success, set_of_rel_paths). Failed workers return (False, set())."""
    rank = worker["rank"]
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        sock.connect((worker["ip"], worker["port"]))
        sock.settimeout(None)
        send_message(sock, ("sync", rank, extensions))
        result = receive_message(sock)
        sock.close()
        return True, set(result) if result else set()
    except Exception as e:
        logger.warning("Sync failed for rank %d: %s", rank, e)
        return False, set()


def _sync_all_workers(workers: list, extensions: list[str]) -> set[str]:
    """Return rel_paths present on ALL reachable workers (intersection).
    Unreachable workers are skipped — their absence doesn't poison the result."""
    per_worker: list[set] = []
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {pool.submit(_sync_worker, w, extensions): w for w in workers}
        for f in as_completed(futures):
            ok, paths = f.result()
            if ok:
                per_worker.append(paths)
    if not per_worker:
        return set()
    result = per_worker[0]
    for s in per_worker[1:]:
        result = result & s
    return result


def _checksum_sync_worker(worker: dict, rel_path: str) -> str:
    """Ask one worker to self-validate its shard for rel_path.
    Returns 'ok', 'mismatch', or 'missing'."""
    rank = worker["rank"]
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        sock.connect((worker["ip"], worker["port"]))
        sock.settimeout(None)
        send_message(sock, ("checksum_sync", rank, rel_path))
        result = receive_message(sock)
        sock.close()
        return result[1] if result else "missing"
    except Exception as e:
        logger.warning("Checksum sync failed for rank %d path %s: %s", rank, rel_path, e)
        return "missing"


def _checksum_sync_all(workers: list, intersection: list[Path], ckpt_root: Path) -> set[Path]:
    """Check all workers for every file in the intersection.
    Returns Path objects where any worker reports mismatch or missing."""
    if not intersection:
        return set()
    corrupted: set[Path] = set()
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {
            pool.submit(_checksum_sync_worker, w, str(p.parent.relative_to(ckpt_root))): (w, p)
            for p in intersection for w in workers
        }
        for f in as_completed(futures):
            worker, path = futures[f]
            status = f.result()
            if status != "ok":
                logger.warning("Checksum %s — rank %d at %s", status, worker["rank"], path.parent.relative_to(ckpt_root))
                corrupted.add(path)
    return corrupted


def _crosscheck_worker(worker: dict, rel_paths: list[str]) -> tuple[int, list[str]]:
    """Ask one worker which rel_paths are missing entirely.
    Returns (rank, missing_rel_paths). Unreachable workers report all as missing."""
    rank = worker["rank"]
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect((worker["ip"], worker["port"]))
        send_message(sock, ("all_shards_present", rank, rel_paths))
        missing = receive_message(sock)
        sock.close()
        return rank, missing if missing else []
    except Exception as e:
        logger.warning("Crosscheck failed for rank %d: %s", rank, e)
        return rank, rel_paths


def _crosscheck_all_workers(workers: list, local_paths: list[Path], ckpt_root: Path, checksum: bool = False) -> list[Path]:
    """Verify every worker has a shard for every local checkpoint.
    If checksum=True, also validates integrity of present shards and includes corrupted ones.
    Returns the subset of local_paths that need re-transfer."""
    if not local_paths:
        return []
    rel_paths = [str(p.parent.relative_to(ckpt_root)) for p in local_paths]
    path_map = {str(p.parent.relative_to(ckpt_root)): p for p in local_paths}
    needs_retry: set[str] = set()
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {pool.submit(_crosscheck_worker, w, rel_paths): w for w in workers}
        for f in as_completed(futures):
            rank, missing = f.result()
            if missing:
                logger.warning("[crosscheck] rank %d missing %d shard(s): %s", rank, len(missing), missing)
                needs_retry.update(missing)
    if checksum:
        present_paths = [p for p in local_paths if str(p.parent.relative_to(ckpt_root)) not in needs_retry]
        corrupted = _checksum_sync_all(workers, present_paths, ckpt_root)
        if corrupted:
            logger.warning("[crosscheck] %d corrupted shard(s) found — queuing for re-transfer", len(corrupted))
            needs_retry.update(str(p.parent.relative_to(ckpt_root)) for p in corrupted)
    return [path_map[rp] for rp in needs_retry if rp in path_map]


def _scan_local(ckpt_root: Path, extensions: list[str]) -> list[Path]:
    """Find all files under ckpt_root matching any of the given extensions."""
    paths = []
    for ext in extensions:
        paths.extend(ckpt_root.rglob(f"*{ext}"))
    return paths
and all th
def _run_pending_loop(pending: list, pending_lock: threading.Lock, trigger: threading.Event) -> None:
    
    """Pending loop: wakes every 10s to re-check pending files for stability."""
    
    while True:
        time.sleep(10)
        if pending:
            logger.info("Re-evaluating %d pending file(s) for stability...", len(pending))
            with pending_lock:
                still_pending, now_stable = [], []
                for path in pending:
                    if not path.exists():
                        logger.info("Pending file no longer exists — dropping: %s", path)
                        continue
                    (now_stable if _is_stable(path) else still_pending).append(path)
                pending[:] = still_pending

            if now_stable:
                logger.info("%d pending file(s) now stable — re-triggering.", len(now_stable))
                trigger.set()
        
def _run_transfer_loop(
    ckpt_root: Path,
    workers: list,
    extensions: list[str],
    trigger: threading.Event,
    pending: list,
    pending_lock: threading.Lock,
) -> None:
    """Transfer thread: wake on trigger, sync, diff, transfer, then check pending.

    checksum_sync (re-hashing existing shards on workers) runs only on the first
    wake-up (startup). Subsequent triggers skip it — per-transfer integrity is
    already guaranteed by the SHA-256 sent with every store_shard call.
    """
    startup = True
    while True:
        trigger.wait()
        trigger.clear()

        # --- file_sync ---
        worker_paths = _sync_all_workers(workers, extensions)
        logger.info("[file_sync] workers have %d path(s)", len(worker_paths))

        local_paths = _scan_local(ckpt_root, extensions)
        intersection = [p for p in local_paths if str(p.parent.relative_to(ckpt_root)) in worker_paths]
        to_transfer  = [p for p in local_paths if str(p.parent.relative_to(ckpt_root)) not in worker_paths]

        # --- checksum_sync (startup only) ---
        # On file-event triggers the per-transfer SHA-256 already guarantees
        # correctness; re-hashing all existing shards every time would block
        # new transfers for minutes when the cluster has many checkpoints.
        checksum_retry: list = []
        if startup:
            logger.info("[checksum_sync] startup integrity sweep — validating %d shared file(s)...", len(intersection))
            corrupted_paths = _checksum_sync_all(workers, intersection, ckpt_root)
            checksum_retry  = list(corrupted_paths)
            startup = False
        else:
            logger.info("[checksum_sync] skipped (file-event trigger — transfer checksum covers integrity)")

        if checksum_retry:
            logger.info("[checksum_retry] re-transferring %d corrupted file(s) first...", len(checksum_retry))
            for path in checksum_retry:
                try:
                    request_store_shards(ckpt_path=str(path), log_fn=logger.info)
                except Exception as e:
                    logger.error("Failed to recover %s: %s", path, e)

        if to_transfer:
            logger.info("[transfer] sending %d missing file(s)...", len(to_transfer))
            for path in to_transfer:
                try:
                    request_store_shards(ckpt_path=str(path), log_fn=logger.info)
                except Exception as e:
                    logger.error("Failed to store %s: %s", path, e)

        if not checksum_retry and not to_transfer:
            logger.info("All files in sync — nothing to transfer.")

        # --- crosscheck: ensure every worker has every shard ---
        logger.info("[crosscheck] verifying all workers have complete shards for %d file(s)...", len(local_paths))
        crosscheck_retry = _crosscheck_all_workers(workers, local_paths, ckpt_root)
        if crosscheck_retry:
            logger.warning("[crosscheck] %d file(s) incomplete on ≥1 worker — re-transferring...", len(crosscheck_retry))
            for path in crosscheck_retry:
                try:
                    request_store_shards(ckpt_path=str(path), log_fn=logger.info)
                except Exception as e:
                    logger.error("Crosscheck re-transfer failed for %s: %s", path, e)
        else:
            logger.info("[crosscheck] all workers complete.")

        if _metrics:
            _metrics.syncs_total.inc()
            _metrics.last_sync.set(time.time())

        # # Re-evaluate pending files
        # with pending_lock:
        #     still_pending, now_stable = [], []
        #     for path in pending:
        #         (now_stable if _is_stable(path) else still_pending).append(path)
        #     pending[:] = still_pending

        # if now_stable:
        #     logger.info("%d pending file(s) now stable — re-triggering.", len(now_stable))
        #     trigger.set()


class CheckpointHandler(FileSystemEventHandler):
    """Watchdog handler: stable files trigger transfer loop; unstable go to pending."""

    def __init__(self, extensions, trigger, pending, pending_lock):
        self._ext = extensions
        self._trigger = trigger
        self._pending = pending
        self._lock = pending_lock

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix not in self._ext:
            return
        logger.info("Detected: %s", path)
        if _is_stable(path):
            self._trigger.set()
        else:
            logger.info("Not yet stable — adding to pending: %s", path)
            with self._lock:
                self._pending.append(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ext",
        default=".safetensors",
        help="Comma-separated file extensions to watch, e.g. .safetensors,.pth",
    )
    args = parser.parse_args()
    extensions = [e.strip() for e in args.ext.split(",")]

    config = _load_config()
    ckpt_root = Path(config["ckpt_root"]).expanduser()
    workers = config["devices_config"]["workers"]
    ckpt_root.mkdir(parents=True, exist_ok=True)

    global _metrics
    _metrics = init_watcher_metrics()
    trigger = threading.Event()
    pending: list = []
    pending_lock = threading.Lock()

    threading.Thread(
        target=_run_transfer_loop,
        args=(ckpt_root, workers, extensions, trigger, pending, pending_lock),
        daemon=True,
    ).start()

    handler = CheckpointHandler(extensions, trigger, pending, pending_lock)
    observer = Observer()
    observer.schedule(handler, str(ckpt_root), recursive=True)
    observer.start()
    logger.info("Watching %s for %s", ckpt_root, extensions)
    logger.info("Waiting 10s for workers to bind before initial sync...")
    time.sleep(10)
    trigger.set()  # initial sync after workers are ready
    
    threading.Thread(
        target=_run_pending_loop,
        args=(pending, pending_lock, trigger),
        daemon=True,
    ).start()
    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
