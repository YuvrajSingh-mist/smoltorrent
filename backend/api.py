import logging
import socket
import sys
import threading
import time
from pathlib import Path
from queue import Queue

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query

sys.path.insert(0, str(Path(__file__).parents[1]))

from networking.send_receive import receive_message, send_message
from utils.common_utils import chunk_data, compute_checksum, load_tensors, merge_shards, model_id_to_dir_name, save_merged_model, save_received_data_shard, shard_from_bytes, shard_to_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SmolTorrent Shard API")

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"
SHARDS_ROOT = Path(__file__).parents[1] / "shards" / "incoming_shards"
MAX_RETRIES = 3

def _load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)



def _send_shard_to_worker(worker: dict, shard: dict, checksum: str) -> tuple[bool, str, dict]:
    """Send one shard to a worker and verify the ack. Returns (ok, error_msg, {})."""
    rank = worker["rank"]
    try:
        sock = _connect_with_retry(worker["ip"], worker["port"], rank)
        send_message(sock, ("store_shard", rank, shard_to_bytes(shard), checksum))
        response = receive_message(sock)
        sock.close()
        if response is None:
            return False, "no response from worker", {}
        command = response[0]
        if command == "store_shard_done":
            _, _, shard_path, metadata_path = response
            logger.info("Worker %d acknowledged shard storage → %s", rank, shard_path)
            return True, "", {"shard_path": shard_path, "metadata_path": metadata_path}
        _, _, err_msg = response
        logger.error("Worker %d store failed: %s", rank, err_msg)
        return False, err_msg, {}
    except Exception as e:
        return False, str(e), {}


def _gather_shard_from_worker(worker: dict, shard: dict, checksum: str) -> tuple[bool, str, dict]:
    """Pull one shard from a worker. Returns (ok, error_msg, result_entry)."""
    rank = worker["rank"]
    host = worker.get("host") or worker.get("device")
    try:
        sock = _connect_with_retry(worker["ip"], worker["port"], rank)
        send_message(sock, ("send_shard", rank))
        shard_bytes = receive_message(sock)
        sock.close()
        if shard_bytes is None:
            return False, "no shard received", {}
        return True, "", {"rank": rank, "host": host, "_shard": shard_from_bytes(shard_bytes)}
    except Exception as e:
        return False, str(e), {}


def _retry_worker(retry_queue: Queue, recovered: list, dead_letter: list, lock: threading.Lock, send_fn) -> None:
    """Daemon thread: drain retry_queue with exponential backoff.

    send_fn is either _send_shard_to_worker or _gather_shard_from_worker.
    Successes append to recovered, permanent failures append to dead_letter.
    """
    while True:
        item = retry_queue.get()
        worker, shard, checksum, attempt = item["worker"], item["shard"], item["checksum"], item["attempt"]
        rank = worker["rank"]
        if attempt > MAX_RETRIES:
            logger.error("Worker %d permanently failed after %d retries", rank, MAX_RETRIES)
            with lock:
                dead_letter.append({"rank": rank, "host": worker.get("host"), "error": "max retries exceeded", "shard_path": None, "metadata_path": None})
            retry_queue.task_done()
            continue
        time.sleep(2 ** attempt)
        ok, err, result = send_fn(worker, shard, checksum)
        if ok:
            with lock:
                recovered.append(result)
        else:
            retry_queue.put({"worker": worker, "shard": shard, "checksum": checksum, "attempt": attempt + 1})
        # retry_queue.task_done()


def _connect_with_retry(ip: str, port: int, rank: int, retries: int = 3, delay: float = 2.0) -> socket.socket:
    for attempt in range(1, retries + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(None)
        try:
            logger.info(f"Connecting to rank {rank} at {ip}:{port} (attempt {attempt}/{retries})")
            sock.connect((ip, port))
            logger.info(f"Connected to rank {rank} at {ip}:{port}")
            return sock
        except (OSError, ConnectionRefusedError) as e:
            sock.close()
            logger.warning(f"Attempt {attempt}/{retries} failed for rank {rank} at {ip}:{port}: {e}")
            if attempt < retries:
                time.sleep(delay)
    raise ConnectionError(f"Could not connect to rank {rank} at {ip}:{port} after {retries} attempts")


@app.post("/gather-shards")
def gather_shards(model_id: str = Query(None, description="HuggingFace model ID, e.g. mlx-community/Qwen2.5-0.5B-Instruct-bf16")):
    config = _load_config()
    workers = config["devices_config"]["workers"]
    if model_id:
        model_name = model_id_to_dir_name(model_id)
    else:
        model_name = Path(config.get("data_path", "model")).parent.name

    logger.info("Gather request for model %s — %d workers", model_name, len(workers))
    gathered: list = []
    dead_letter: list = []
    lock = threading.Lock()
    gather_queue: Queue = Queue()

    threading.Thread(
        target=_retry_worker, args=(gather_queue, gathered, dead_letter, lock, _gather_shard_from_worker), daemon=True
    ).start()

    for worker in workers:
        logger.info("Pulling shard from rank %d (%s)", worker["rank"], worker.get("host") or worker.get("device"))
        ok, err, result = _gather_shard_from_worker(worker, {}, "")
        if ok:
            logger.info("  ✓ rank %d received", worker["rank"])
            with lock:
                gathered.append(result)
        else:
            logger.warning("rank %d gather failed (%s) — queuing for retry", worker["rank"], err)
            gather_queue.put({"worker": worker, "shard": {}, "checksum": "", "attempt": 1})

    logger.info("All pulls dispatched — waiting for retry queue to drain")
    gather_queue.join()

    with lock:
        all_gathered = list(gathered)
        failed = list(dead_letter)

    if failed:
        raise HTTPException(
            status_code=500,
            detail={"gathered": all_gathered, "errors": failed},
        )

    logger.info("All %d shards gathered — saving to disk", len(all_gathered))
    errors = []
    for entry in all_gathered:
        shard_path, _, ok, err = save_received_data_shard(
            shard=entry["_shard"],
            metadata={"rank": entry["rank"], "role": "gathered", "host": entry["host"]},
            output_dir=SHARDS_ROOT / model_name / f"worker-{entry['rank']}",
        )
        if not ok:
            errors.append({"rank": entry["rank"], "host": entry["host"], "error": err})
        else:
            entry["shard_path"] = shard_path
            logger.info("  rank %d saved → %s", entry["rank"], shard_path)

    if errors:
        raise HTTPException(status_code=500, detail={"gathered": all_gathered, "errors": errors})

    save_path = config.get("save_path")
    logger.info("Merging %d shards → %s", len(all_gathered), save_path)
    merged = merge_shards([entry["_shard"] for entry in all_gathered])
    save_merged_model(merged, save_path)
    logger.info("Merged model saved → %s", save_path)

    for entry in all_gathered:
        entry.pop("_shard", None)

    return {"gathered": all_gathered, "save_path": save_path}


@app.post("/store-shard")
def store_shard(
    model_id: str = Query(None, description="HuggingFace model ID; if omitted, derived from config data_path"),
):
    """Load the model from config data_path, shard it, and push each shard to its ranked Pi worker via TCP."""
    config = _load_config()
    workers = config["devices_config"]["workers"]
    num_workers = len(workers)

    if model_id:
        model_name = model_id_to_dir_name(model_id)
    else:
        model_name = Path(config.get("data_path", "model")).parent.name

    data_path = Path(config["data_path"])
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"data_path not found: {data_path}")

    logger.info("Loading tensors from %s", data_path)
    tensors = load_tensors(data_path)
    total_mb = sum(v.nbytes for v in tensors.values()) / 1024**2
    logger.info("Loaded %d tensors (%.1f MB) — chunking into %d shards", len(tensors), total_mb, num_workers)
    chunks = chunk_data(tensors, n_chunks=num_workers)
    logger.info("Chunking done — serializing and sending to workers")

    sent: list = []
    dead_letter: list = []
    lock = threading.Lock()
    store_queue: Queue = Queue()

    threading.Thread(
        target=_retry_worker, args=(store_queue, sent, dead_letter, lock, _send_shard_to_worker), daemon=True
    ).start()

    for i, worker in enumerate(workers):
        shard = chunks[i]
        logger.info("Serializing shard for rank %d (%d tensors)", worker["rank"], len(shard))
        shard_bytes = shard_to_bytes(shard)
        checksum = compute_checksum(shard_bytes)
        logger.info("Sending shard to rank %d (%s) — %.2f MB", worker["rank"], worker.get("host") or worker.get("device"), len(shard_bytes) / 1024**2)
        ok, err, result = _send_shard_to_worker(worker, shard, checksum)
        if ok:
            with lock:
                sent.append({"rank": worker["rank"], "host": worker.get("host") or worker.get("device"), "shard_path": result.get("shard_path"), "metadata_path": result.get("metadata_path")})
        else:
            logger.warning("rank %d initial send failed (%s) — queuing for retry", worker["rank"], err)
            store_queue.put({"worker": worker, "shard": shard, "checksum": checksum, "attempt": 1})

    logger.info("All sends dispatched — waiting for retry queue to drain")
    store_queue.join()

    with lock:
        failed = list(dead_letter)
        succeeded = list(sent)

    if failed:
        raise HTTPException(
            status_code=500,
            detail={"sent": succeeded, "permanently_failed": failed},
        )

    logger.info("Store complete — %d/%d shards sent successfully", len(succeeded), num_workers)
    return {"model_name": model_name, "num_shards": num_workers, "sent_to": succeeded}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
