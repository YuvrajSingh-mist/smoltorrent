"""FastAPI server that orchestrates shard distribution across workers.

Exposes two endpoints:
  POST /store-shard  — shards a checkpoint and pushes each shard to its ranked worker.
  POST /gather-shards — pulls shards from all workers, saves locally, then merges.
"""

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).parents[1]))

from networking.send_receive import network_metrics
from utils.common_utils import (
    chunk_data,
    compute_checksum,
    load_config,
    load_tensors,
    merge_shards,
    save_merged_model,
    save_shard,
    shard_to_bytes,
)
from utils.network_metrics import log_network_metrics
from utils.prometheus_utils import (
    make_asgi_app,
    api_store_ops,
    api_gather_ops,
    api_xfer_errors,
    api_store_wall,
    api_gather_wall,
)
from utils.worker_ops import (
    gather_shard_from_worker,
    heartbeat_workers,
    run_retry_worker,
    send_shard_to_worker,
)
from discovery import discover_workers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SmolTorrent Shard API")
app.mount("/metrics", make_asgi_app())

SHARDS_ROOT = Path(__file__).parents[1] / "shards"
REDUNDANCY = 2  # replicas per shard (1 = no redundancy, 2 = one primary + one replica)


def _init_error_labels() -> None:
    cfg = load_config()
    for w in cfg["devices_config"]["workers"]:
        api_xfer_errors.labels(rank=str(w["rank"]))


try:
    _init_error_labels()
except Exception:
    pass  # config missing at import time — labels register on first use


def _log(msg: str) -> str:
    logger.info("[api] %s", msg)
    return msg + "\n"


@app.post("/store-shard")
def store_shard(
    ckpt_path: str = Query(..., description="Absolute path to the checkpoint .safetensors file on master"),
):
    """Load a checkpoint, shard it, push each shard to its ranked worker, stream log lines."""

    def _generate():
        config = load_config()
        workers = config["devices_config"]["workers"]
        num_workers = len(workers)

        yield _log(f"Heartbeat: checking {num_workers} worker(s)…")
        dead = heartbeat_workers(workers)
        if dead:
            names = ", ".join(f"rank {d['rank']} ({d['host']})" for d in dead)
            yield f"ERROR: {len(dead)} worker(s) unreachable: {names}\n"
            return
        yield _log("Heartbeat: all workers alive")

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

        store_start = time.monotonic()
        tensors = load_tensors(ckpt_file)
        total_mb = sum(v.nbytes for v in tensors.values()) / 1024**2
        yield _log(f"Loaded {len(tensors)} tensors ({total_mb:.1f} MB) from {rel_path} — chunking into {num_workers} shards")
        chunks = chunk_data(tensors, n_chunks=num_workers)

        sent: list = []
        dead_letter: list = []
        lock = threading.Lock()
        store_queue: Queue = Queue()

        threading.Thread(target=run_retry_worker, args=(store_queue, sent, dead_letter, lock), daemon=True).start()

        # Serialize all shards up front (decoupled from worker assignment)
        shards = [(shard_to_bytes(chunks[i]), None) for i in range(num_workers)]
        shards = [(sb, compute_checksum(sb)) for sb, _ in shards]

        # Round 0: shard i → workers[i]  (primary)
        # Round 1: shard i → workers[(i+1) % n]  (replica)
        # Sequential rounds prevent two concurrent 235 MB receives on the same Pi.
        for round_idx in range(REDUNDANCY):
            round_jobs = [
                (workers[(i + round_idx) % num_workers], sb, cs)
                for i, (sb, cs) in enumerate(shards)
            ]
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                future_to_job = {
                    pool.submit(send_shard_to_worker, worker, shard_bytes, checksum, rel_path): (worker, shard_bytes, checksum)
                    for worker, shard_bytes, checksum in round_jobs
                }
                for future in as_completed(future_to_job):
                    worker, shard_bytes, checksum = future_to_job[future]
                    rank = worker["rank"]
                    host = worker.get("host") or worker.get("device")
                    ok, err, result = future.result()
                    if ok:
                        with lock:
                            sent.append({"rank": rank, "host": host, "shard_path": result.get("shard_path")})
                        yield _log(f"  ✓ rank {rank} ({host}) [round {round_idx}]")
                    else:
                        yield _log(f"  ↻ rank {rank} ({host}) failed — queuing retry: {err}")
                        store_queue.put({
                            "fn": lambda w=worker, sb=shard_bytes, cs=checksum: send_shard_to_worker(w, sb, cs, rel_path),
                            "worker": worker,
                            "attempt": 1,
                        })

        store_queue.join()

        failed = list(dead_letter)
        succeeded = list(sent)

        for f in failed:
            api_xfer_errors.labels(rank=str(f["rank"])).inc()
            yield _log(f"  ✗ rank {f['rank']} ({f.get('host')}) permanently failed: {f.get('error')}")

        total_expected = num_workers * REDUNDANCY
        log_network_metrics(network_metrics.get_metrics(), logger, "store")
        api_store_wall.observe(time.monotonic() - store_start)
        if failed:
            yield f"ERROR: {len(failed)}/{total_expected} sends failed\n"
        else:
            api_store_ops.inc()
            yield _log(f"Done: {len(succeeded)}/{total_expected} sends ({REDUNDANCY}x replicated) → {rel_path}")

    return StreamingResponse(_generate(), media_type="text/plain")


@app.post("/gather-shards")
def gather_shards(
    ckpt_path: str = Query(..., description="Absolute path to the checkpoint file (same path used for store)"),
):
    """Pull shards from every worker, save each as it arrives, merge, stream log lines."""

    def _generate():
        config = load_config()
        workers = config["devices_config"]["workers"]
        num_workers = len(workers)

        yield _log(f"Heartbeat: checking {num_workers} worker(s)…")
        dead = heartbeat_workers(workers)
        if dead:
            names = ", ".join(f"rank {d['rank']} ({d['host']})" for d in dead)
            yield f"ERROR: {len(dead)} worker(s) unreachable: {names}\n"
            return
        yield _log("Heartbeat: all workers alive")

        ckpt_root = Path(config["ckpt_root"]).expanduser()
        ckpt_file = Path(ckpt_path).expanduser()

        try:
            rel_path = str(ckpt_file.parent.relative_to(ckpt_root))
        except ValueError:
            yield f"ERROR: {ckpt_file} is not under ckpt_root {ckpt_root}\n"
            return

        gather_start = time.monotonic()

        gathered: list = []
        # Keyed by shard index (0..n-1), not worker rank — matters when a replica
        # serves a shard so it lands in the correct merge slot.
        shards_by_index: dict = {}
        save_errors: list = []
        dead_letter: list = []
        lock = threading.Lock()
        gather_queue: Queue = Queue()

        def _gather_and_save(worker: dict, shard_index: int) -> tuple[bool, str, dict]:
            ok, err, result = gather_shard_from_worker(worker, rel_path)
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

        threading.Thread(target=run_retry_worker, args=(gather_queue, gathered, dead_letter, lock), daemon=True).start()

        def _gather_one(i: int, worker: dict):
            rank = worker["rank"]
            host = worker.get("host") or worker.get("device")
            ok, err, result = _gather_and_save(worker, shard_index=i)
            if not ok and REDUNDANCY > 1:
                replica = workers[(i + 1) % num_workers]
                ok, err, result = _gather_and_save(replica, shard_index=i)
                if not ok:
                    return i, rank, host, False, err, result
                return i, replica["rank"], replica.get("host") or replica.get("device"), True, "", result
            return i, rank, host, ok, err, result

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            future_to_idx = {pool.submit(_gather_one, i, w): i for i, w in enumerate(workers)}
            for future in as_completed(future_to_idx):
                i, rank, host, ok, err, result = future.result()
                if ok:
                    gathered.append(result)
                    yield _log(f"  ✓ shard {i} — saved → {result['shard_path']}")
                else:
                    yield _log(f"  ↻ shard {i} (rank {rank}) failed — queuing retry: {err}")
                    worker = workers[i]
                    gather_queue.put({
                        "fn": lambda w=worker, si=i: _gather_and_save(w, si),
                        "worker": worker,
                        "attempt": 1,
                    })

        gather_queue.join()

        all_gathered = list(gathered)
        failed = list(dead_letter) + list(save_errors)

        if failed:
            for f in failed:
                api_xfer_errors.labels(rank=str(f["rank"])).inc()
                yield _log(f"  ✗ rank {f['rank']} ({f.get('host')}): {f['error']}")
            yield f"ERROR: {len(failed)}/{num_workers} shards failed — skipping merge\n"
            return

        save_path = Path(config["ckpt_root"]).expanduser() / rel_path / "merged.safetensors"
        yield _log(f"Merging {len(all_gathered)} shards → {save_path}")
        merged = merge_shards([shards_by_index[i] for i in range(num_workers)])
        save_merged_model(merged, save_path)
        log_network_metrics(network_metrics.get_metrics(), logger, "gather")
        api_gather_wall.observe(time.monotonic() - gather_start)
        api_gather_ops.inc()
        yield _log(f"Done: saved → {save_path}")

    return StreamingResponse(_generate(), media_type="text/plain")


@app.get("/discover")
def discover(timeout: float = Query(10.0, description="How long to scan for workers (seconds)")):
    """Scan the local network for smoltorrent worker nodes via mDNS."""
    workers = discover_workers(timeout=timeout)
    logger.info("[api] Discovery found %d worker(s): %s", len(workers), workers)
    return {"workers": workers}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
