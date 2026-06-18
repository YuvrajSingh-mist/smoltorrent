from __future__ import annotations

"""FastAPI server that orchestrates shard distribution across workers.

Exposes two endpoints:
  POST /store-shard  — shards a checkpoint and pushes each shard to its ranked worker.
  POST /gather-shards — pulls shards from all workers, saves locally, then merges.
"""

import json 
import logging
import mmap
import os
import tempfile
import struct
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
    compute_checksum,
    dir_name_to_model_id,
    fetch_model_metadata,
    get_shard_ranges,
    handle_json_header,
    load_config,
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
    gather_shard_data_only,
    heartbeat_workers,
    run_retry_worker,
    send_shard_to_worker,
)
from discovery import discover_workers
from utils.shard_tracker import add_shard_header, get_ranks, get_shard_header, get_replica_map, list_all_shard_headers
from utils.observability import setup_api

import socket as _socket
setup_api(hostname=_socket.gethostname())

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
    """Pre-initialise Prometheus transfer-error label combinations at import time.

    Args:
        None: reads worker list from configs/config.yaml via load_config().

    Returns:
        None.
    """
    cfg = load_config()
    for w in cfg["devices_config"]["workers"]:
        api_xfer_errors.labels(rank=str(w["rank"]))


try:
    _init_error_labels()
except Exception:
    pass  # config missing at import time — labels register on first use


def _log(msg: str) -> str:
    """Log *msg* at INFO and return it as a streaming-response line.

    Args:
        msg: Human-readable status message to log and stream to the client.

    Returns:
        The message string with a trailing newline appended.
    """
    logger.info("[api] %s", msg)
    return msg + "\n"


@app.post("/store-shard")
def store_shard(
    ckpt_path: str = Query(..., description="Absolute path to the checkpoint .safetensors file on master"),
):
    """Load a checkpoint, shard it, push each shard to its ranked worker, stream log lines.

    Args:
        ckpt_path: Absolute path to the checkpoint .safetensors file on the master node.

    Returns:
        StreamingResponse of ``text/plain`` lines — each line is a status message.
        Lines beginning with ``ERROR:`` indicate a fatal failure.
    """

    def _generate():
        """Generator that performs the shard store and yields streaming log lines.

        Args:
            None: captures ``ckpt_path`` from the enclosing scope.

        Returns:
            Generator of ``str`` log lines for :class:`~fastapi.responses.StreamingResponse`.
        """
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
            logger.error("[api] store_shard: checkpoint not found: %s", ckpt_file)
            yield f"ERROR: checkpoint not found: {ckpt_file}\n"
            return

        try:
            rel_path = str(ckpt_file.parent.relative_to(ckpt_root))
        except ValueError:
            logger.error("[api] store_shard: %s is not under ckpt_root %s", ckpt_file, ckpt_root)
            yield f"ERROR: {ckpt_file} is not under ckpt_root {ckpt_root}\n"
            return

        store_start = time.monotonic()
        # Parse just the JSON header — no tensor data loaded into memory
        header, data_section_offset = handle_json_header(str(ckpt_file))
        shard_ranges, shard_tensor_meta = get_shard_ranges(header, data_section_offset, num_workers)
        total_tensor_bytes = sum(r["length"] for r in shard_ranges)
        total_mb = total_tensor_bytes / 1024**2
        yield _log(f"Parsed header from {rel_path} ({total_mb:.1f} MB tensor data) — {num_workers} shards")

        # Compute original checksum now — same 1 MB streaming pass, warms page cache
        # for the subsequent sendfile passes (OS page cache bridge).
        yield _log("Computing original tensor checksum…")
        original_checksum = compute_checksum(str(ckpt_file), offset=data_section_offset, length=total_tensor_bytes)
        yield _log(f"Original checksum: {original_checksum[:16]}…")

        add_shard_header(
            shard_key=rel_path,
            header_json=json.dumps(header, separators=(",", ":")),
            data_section_offset=data_section_offset,
            num_workers=num_workers,
            shard_ranges=shard_ranges,
            total_tensor_bytes=total_tensor_bytes,
            original_checksum=original_checksum,
        )

        sent: list = []
        dead_letter: list = []
        lock = threading.Lock()
        store_queue: Queue = Queue()

        logger.info("[api] store_shard: rel_path=%s num_workers=%d total_mb=%.1f",
                    rel_path, num_workers, sum(r["length"] for r in shard_ranges) / 1024**2)
        threading.Thread(target=run_retry_worker, args=(store_queue, sent, dead_letter, lock), daemon=True).start()

        ckpt_size = ckpt_file.stat().st_size

        # Round 0: shard i → workers[i]          saved as shard_0.safetensors (primary)
        # Round 1: shard i → workers[(i+1) % n]   saved as shard_1.safetensors (replica)
        # send_shard_to_worker writes to the tracker DB immediately on worker ack,
        # so retries that succeed also get recorded without extra bookkeeping here.
        for round_idx in range(REDUNDANCY):
            shard_filename = f"shard_{round_idx}.safetensors"
            round_jobs = [
                (i, workers[(i + round_idx) % num_workers], shard_ranges[i], shard_tensor_meta[i])
                for i in range(num_workers)
            ]
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                future_to_job = {
                    pool.submit(
                        send_shard_to_worker,
                        worker, str(ckpt_file), shard_range["file_offset"], shard_range["length"],
                        tensor_meta, rel_path, shard_filename,
                        shard_idx, ckpt_size, str(ckpt_file),
                    ): (shard_idx, worker)
                    for shard_idx, worker, shard_range, tensor_meta in round_jobs
                }
                for future in as_completed(future_to_job):
                    shard_idx, worker = future_to_job[future]
                    rank = worker["rank"]
                    host = worker.get("host") or worker.get("device")
                    ok, err, result = future.result()
                    if ok:
                        with lock:
                            sent.append({"rank": rank, "host": host, "shard_path": result.get("shard_path")})
                        yield _log(f"  ✓ rank {rank} ({host}) [shard_index={shard_idx} {shard_filename}]")
                    else:
                        yield _log(f"  ↻ rank {rank} ({host}) failed — queuing retry: {err}")
                        store_queue.put({
                            "fn": lambda w=worker, sr=shard_ranges[shard_idx], tm=shard_tensor_meta[shard_idx],
                                         sf=shard_filename, si=shard_idx: send_shard_to_worker(
                                w, str(ckpt_file), sr["file_offset"], sr["length"], tm, rel_path, sf,
                                si, ckpt_size, str(ckpt_file),
                            ),
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
            yield _log(f"Shard key: {rel_path}")

    return StreamingResponse(_generate(), media_type="text/plain")


@app.get("/models")
def list_models(
    model:    str | None = Query(None, description="Filter by model name"),
    variant:  str | None = Query(None, description="Filter by variant"),
    step_min: int | None = Query(None, description="Filter by step >= step_min"),
    step_max: int | None = Query(None, description="Filter by step <= step_max"),
    since:    str | None = Query(None, description="Filter by stored_at >= since (ISO date, e.g. 2026-06-01)"),
    until:    str | None = Query(None, description="Filter by stored_at <= until (ISO date)"),
    limit:    int        = Query(50,   description="Page size (max 200)"),
    page:     int        = Query(1,    description="1-based page number"),
):
    """List stored checkpoints with optional filtering and pagination.

    Args:
        model:    Filter by model_name (exact match).
        variant:  Filter by variant (exact match).
        step_min: Return only checkpoints with step >= step_min.
        step_max: Return only checkpoints with step <= step_max.
        since:    Return only checkpoints stored on or after this ISO date.
        until:    Return only checkpoints stored on or before this ISO date.
        limit:    Maximum results per page (default 50, capped at 200).
        page:     1-based page number (default 1).

    Returns:
        Dict with keys ``models`` (list of checkpoint summaries), ``total``,
        ``page``, ``limit``, and ``pages``.
    """
    return list_all_shard_headers(
        model=model, variant=variant,
        step_min=step_min, step_max=step_max,
        since=since, until=until,
        limit=limit, page=page,
    )


@app.get("/models/{model_name}")
def list_model_history(
    model_name: str,
    variant:  str | None = Query(None),
    step_min: int | None = Query(None),
    step_max: int | None = Query(None),
    since:    str | None = Query(None),
    until:    str | None = Query(None),
    limit:    int        = Query(50),
    page:     int        = Query(1),
):
    """List all stored checkpoints for a specific model, newest first.

    Args:
        model_name: Model name to filter by (path parameter).
        variant:    Optional variant filter (exact match).
        step_min:   Return only checkpoints with step >= step_min.
        step_max:   Return only checkpoints with step <= step_max.
        since:      Return only checkpoints stored on or after this ISO date.
        until:      Return only checkpoints stored on or before this ISO date.
        limit:      Maximum results per page (default 50, capped at 200).
        page:       1-based page number (default 1).

    Returns:
        Dict with keys ``models``, ``total``, ``page``, ``limit``, and ``pages``.
    """
    return list_all_shard_headers(
        model=model_name, variant=variant,
        step_min=step_min, step_max=step_max,
        since=since, until=until,
        limit=limit, page=page,
    )


@app.post("/gather-shards")
def gather_shards(
    shard_key: str = Query(..., description="Shard key returned by /store-shard, e.g. 'Qwen2.5-0.5B/base/step_0'"),
):
    """Stream tensor bytes from workers directly into a pre-allocated merged safetensors file.

    No tensors are loaded into Python memory on the coordinator.  Each worker sends only
    its raw tensor data bytes (stripped of its local safetensors header); the coordinator
    writes them at the correct byte offset of the merged file using receive_into_fd_offset.

    Args:
        shard_key: Relative checkpoint path returned by ``/store-shard``,
                   e.g. ``"Qwen2.5-0.5B/base/step_0"``.

    Returns:
        StreamingResponse of ``text/plain`` lines — each line is a status message.
        Lines beginning with ``ERROR:`` indicate a fatal failure.
    """

    def _generate():
        """Generator that performs the shard gather and yields streaming log lines.

        Args:
            None: captures ``shard_key`` from the enclosing scope.

        Returns:
            Generator of ``str`` log lines for :class:`~fastapi.responses.StreamingResponse`.
        """
        config = load_config()
        workers = config["devices_config"]["workers"]
        num_workers = len(workers)

        yield _log(f"Heartbeat: checking {num_workers} worker(s)…")
        dead = heartbeat_workers(workers)
        if dead:
            names = ", ".join(f"rank {d['rank']} ({d['host']})" for d in dead)
            yield _log(f"Warning: {len(dead)} worker(s) unreachable: {names} — attempting replica fallback")
        else:
            yield _log("Heartbeat: all workers alive")

        rel_path = shard_key

        # Retrieve stored header — required for zero-copy streaming merge.
        stored = get_shard_header(rel_path)
        if not stored:
            yield f"ERROR: no header stored for {rel_path} — run store-shard first\n"
            return

        header = json.loads(stored["header_json"])
        data_section_offset = stored["data_section_offset"]
        stored_num_workers = stored["num_workers"]

        # Use precomputed shard_ranges if available — skips re-sorting N tensors.
        if stored["shard_ranges"]:
            shard_ranges = stored["shard_ranges"]
            total_tensor_bytes = stored["total_tensor_bytes"]
            yield _log(f"Using cached shard ranges ({total_tensor_bytes / 1024**2:.1f} MB)")
        else:
            shard_ranges, _ = get_shard_ranges(header, data_section_offset, stored_num_workers)
            total_tensor_bytes = sum(r["length"] for r in shard_ranges)
            yield _log(f"Recomputed shard ranges ({total_tensor_bytes / 1024**2:.1f} MB — upgrade: re-store to cache)")

        # Consult the tracker to avoid asking workers that don't hold this shard.
        tracked_ranks = get_ranks(rel_path)
        if tracked_ranks:
            gather_workers = [w for w in workers if w["rank"] in tracked_ranks]
            yield _log(f"Tracker: {len(gather_workers)}/{len(workers)} worker(s) known to hold this shard")
        else:
            gather_workers = workers
            yield _log("Tracker: no entry — broadcasting to all workers")

        if len(gather_workers) != stored_num_workers:
            logger.error(
                "[api] gather_shards: inconsistency rel_path=%s tracker_workers=%d header_num_workers=%d",
                rel_path, len(gather_workers), stored_num_workers,
            )
            yield f"ERROR: tracker has {len(gather_workers)} worker(s) but header says {stored_num_workers} — data inconsistency\n"
            return

        gather_start = time.monotonic()

        # Pre-allocate merged file: [uint64 hdr_len][JSON header][tensor bytes...]
        
        save_path = Path(config["ckpt_root"]).expanduser() / rel_path / "merged.safetensors"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        header_json_bytes = stored["header_json"].encode()
        merged_header_size = 8 + len(header_json_bytes)
        total_file_size = merged_header_size + total_tensor_bytes

        temp_save_file = tempfile.NamedTemporaryFile(dir=save_path.parent, prefix="gather_", suffix=".tmp", delete=False)
        temp_save_path = Path(temp_save_file.name)
        
        yield _log(f"Pre-allocating merged file ({total_file_size / 1024**2:.1f} MB) → {save_path}")
        with open(temp_save_path, "wb") as mf:
            mf.write(struct.pack("<Q", len(header_json_bytes)))
            mf.write(header_json_bytes)
            _fallocate = getattr(os, "posix_fallocate", None)
            if _fallocate is not None:
                try:
                    _fallocate(mf.fileno(), 0, total_file_size)
                except OSError:
                    mf.truncate(total_file_size)
            else:
                mf.truncate(total_file_size)

        gathered: list = []
        dead_letter: list = []
        lock = threading.Lock()
        gather_queue: Queue = Queue()

        # Fetch the complete redundancy map upfront — one DB query for all shard indices.
        # Each entry: replica_map[i] = [{"rank": r, "shard_file": f}, ...] primary-first.
        replica_map = get_replica_map(rel_path)
        worker_by_rank = {w["rank"]: w for w in workers}
        logger.info("[api] gather replica_map: %s", {i: [(r["rank"], r["shard_file"]) for r in v] for i, v in replica_map.items()})

        threading.Thread(target=run_retry_worker, args=(gather_queue, gathered, dead_letter, lock), daemon=True).start()

            
        with open(temp_save_path, "r+b") as mf:
            with mmap.mmap(mf.fileno(), length=total_file_size, access=mmap.ACCESS_WRITE) as merged_mm:

                def _stream_one(shard_index: int) -> tuple[bool, str, dict]:
                    """Fetch shard *shard_index* from its best available replica into the merged mmap.

                    Tries replicas in primary-first order; falls through on failure or checksum
                    mismatch.  Thread-safe when called concurrently with non-overlapping shard
                    indices.

                    Args:
                        shard_index: Zero-based index of the shard to retrieve.

                    Returns:
                        Tuple of ``(ok, error_msg, result_dict)``.  On success ``result_dict``
                        contains ``shard_index``, ``rank``, ``host``, and ``checksum``.
                    """
                    write_offset = merged_header_size + (shard_ranges[shard_index]["file_offset"] - data_section_offset)
                    data_length = shard_ranges[shard_index]["length"]

                    replicas = replica_map.get(shard_index)
                    if not replicas:
                        # Legacy store: no shard_index recorded. Fall back positionally.
                        logger.warning("[api] shard %d: no replica info in tracker — using positional fallback", shard_index)
                        replicas = [{"rank": gather_workers[shard_index]["rank"], "shard_file": "shard_0.safetensors", "checksum": ""}]
                        if REDUNDANCY > 1:
                            replicas.append({"rank": workers[(shard_index + 1) % num_workers]["rank"], "shard_file": "shard_1.safetensors", "checksum": ""})

                    ok, err = False, "no replicas tried"
                    rank, host, stored_checksum = -1, "unknown", ""
                    for rep in replicas:
                        rep_worker = worker_by_rank.get(rep["rank"])
                        if rep_worker is None:
                            logger.warning("[api] shard %d: rank %d in tracker but missing from config — skipping", shard_index, rep["rank"])
                            continue
                        rank = rep_worker["rank"]
                        host = rep_worker.get("host") or rep_worker.get("device", "")
                        logger.info("[api] shard %d: trying rank %d (%s) file=%s", shard_index, rank, host, rep["shard_file"])
                        ok, err = gather_shard_data_only(
                            rep_worker, rel_path, merged_mm, write_offset, data_length,
                            shard_filename=rep["shard_file"],
                        )
                        if ok:
                            stored_checksum = rep.get("checksum", "")
                            if not stored_checksum:
                                logger.warning("[api] shard %d: rank %d (%s) has no stored checksum — skipping integrity check", shard_index, rank, host)
                                break
                            actual = compute_checksum(temp_save_path, offset=write_offset, length=data_length)
                            if actual == stored_checksum:
                                break
                            # Checksum mismatch — this replica's on-disk shard is corrupt.
                            # Fall through to try the next replica (it overwrites the same offset).
                            logger.warning(
                                "[api] shard %d: rank %d checksum mismatch — expected %s… got %s… — trying next replica",
                                shard_index, rank, stored_checksum[:16], actual[:16],
                            )
                            ok, err = False, f"checksum mismatch from rank {rank}"
                        else:
                            logger.warning("[api] shard %d: rank %d failed (%s) — trying next replica", shard_index, rank, err)

                    return ok, err, {"shard_index": shard_index, "rank": rank, "host": host, "checksum": stored_checksum}

                with ThreadPoolExecutor(max_workers=stored_num_workers) as pool:
                    futures = {pool.submit(_stream_one, i): i for i in range(stored_num_workers)}
                    for future in as_completed(futures):
                        ok, err, result = future.result()
                        shard_index, rank, host, stored_checksum = result["shard_index"], result["rank"], result["host"], result["checksum"]
                        if ok:
                            with lock:
                                gathered.append(result)
                            cs = f" [{stored_checksum[:16]}…]" if stored_checksum else " [no checksum]"
                            yield _log(f"  ✓ shard {shard_index} — rank {rank} ({host}){cs}")
                        else:
                            # _stream_one already exhausted every replica — retrying is pointless.
                            # Record as permanently failed immediately.
                            n_replicas = len(replica_map.get(shard_index) or [])
                            yield _log(
                                f"  ✗ shard {shard_index} — all {n_replicas} replica(s) exhausted: {err}"
                            )
                            with lock:
                                dead_letter.append({
                                    "rank": rank, "host": host,
                                    "error": err, "shard_index": shard_index,
                                })

                # Signal the retry worker there is nothing queued so join() returns.
                gather_queue.join()
                merged_mm.flush()
                
        os.replace(temp_save_path, save_path)
        temp_save_file.close()
        failed = list(dead_letter)

        if failed:
            save_path.unlink(missing_ok=True)
            details = "; ".join(
                f"shard {f['shard_index']} (rank {f['rank']} / {f.get('host', '?')}): {f.get('error', '?')}"
                for f in failed
            )
            for f in failed:
                api_xfer_errors.labels(rank=str(f["rank"])).inc()
            yield f"ERROR: {len(failed)}/{stored_num_workers} shard(s) permanently failed — merged file deleted\n"
            yield f"ERROR: details — {details}\n"
            return

        # Final end-to-end check: merged tensor section must match the original file.
        original_checksum = stored.get("original_checksum", "")
        if original_checksum:
            yield _log("Verifying merged file against original checksum…")
            merged_checksum = compute_checksum(str(save_path), offset=merged_header_size, length=total_tensor_bytes)
            if merged_checksum == original_checksum:
                yield _log(f"✓ Integrity verified — {original_checksum[:16]}…")
            else:
                logger.error(
                    "[api] gather FINAL integrity FAIL: expected=%s got=%s",
                    original_checksum[:16], merged_checksum[:16],
                )
                save_path.unlink(missing_ok=True)
                yield f"ERROR: merged file checksum mismatch — expected {original_checksum[:16]}… got {merged_checksum[:16]}…\n"
                return
        else:
            yield _log("~ No original checksum stored (legacy store) — skipping final integrity check")

        log_network_metrics(network_metrics.get_metrics(), logger, "gather")
        api_gather_wall.observe(time.monotonic() - gather_start)
        api_gather_ops.inc()
        yield _log(f"Done: {len(gathered)}/{stored_num_workers} shards → {save_path}")

        model_dir_name = Path(rel_path).parts[0]
        model_id = dir_name_to_model_id(model_dir_name)
        yield _log(f"Fetching tokenizer and config for {model_id} from HuggingFace Hub...")
        try:
            fetch_model_metadata(model_id, config)
        except Exception as e:
            logger.error("[api] fetch_model_metadata failed for %s: %s", model_id, e, exc_info=True)
            yield f"Warning: metadata fetch failed for {model_id}: {e}\n"
        yield _log(f"Model directory ready for inference → {save_path.parent}")

    return StreamingResponse(_generate(), media_type="text/plain")


@app.get("/discover")
def discover(timeout: float = Query(10.0, description="How long to scan for workers (seconds)")):
    """Scan the local network for smoltorrent worker nodes via mDNS.

    Args:
        timeout: How long to listen for mDNS announcements in seconds (default 10.0).

    Returns:
        Dict with key ``workers`` — a list of discovered worker dicts, each containing
        ``ip``, ``port``, ``rank``, and ``hostname``.
    """
    workers = discover_workers(timeout=timeout)
    logger.info("[api] Discovery found %d worker(s): %s", len(workers), workers)
    return {"workers": workers}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
