import logging
import socket
import sys
import time
from io import BytesIO
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from safetensors.torch import load as st_load

sys.path.insert(0, str(Path(__file__).parents[1]))

from networking.send_receive import receive_message, send_message
from utils.common_utils import merge_shards, save_full_model, save_received_data_shard

logger = logging.getLogger(__name__)

app = FastAPI(title="SmolTorrent Shard API")

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


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
def gather_shards():
    config = _load_config()
    workers = config["devices_config"]["workers"]
    shard_save_root = config.get("received_shards_dir", "shards/incoming_shards")

    gathered = []
    errors = []

    for worker in workers:
        host = worker.get("host") or worker.get("device")
        ip = worker["ip"]
        port = worker["port"]
        rank = worker["rank"]

        try:
            sock = _connect_with_retry(ip, port, rank)
            send_message(sock, ("send_shard", rank))
            shard = receive_message(sock)
            sock.close()

            shard_path, _ = save_received_data_shard(
                shard=shard,
                metadata={"rank": rank, "role": "gathered", "host": host},
                output_dir=f"{shard_save_root}/gathered/from-rank-{rank}",
            )
            gathered.append({"rank": rank, "host": host, "shard_path": shard_path, "_shard": shard})

        except Exception as e:
            errors.append({"rank": rank, "host": host, "error": str(e)})

    if errors:
        raise HTTPException(
            status_code=500,
            detail={"gathered": gathered, "errors": errors},
        )

    # Merge all shards and save the full model to save_path
    save_path = config.get("save_path")
    source_model_dir = Path(config["data_path"]).expanduser().parent
    all_shards = [entry["_shard"] for entry in gathered]
    merged = merge_shards(all_shards)
    save_full_model(merged, source_model_dir, save_path)
    logger.info("Full model saved → %s", save_path)

    # Strip internal _shard key before returning
    for entry in gathered:
        entry.pop("_shard", None)

    return {"gathered": gathered, "save_path": save_path}


@app.post("/store-shard")
async def store_shard(
    rank: int = Form(...),
    role: str = Form("received"),
    host: str = Form(""),
    output_dir: str = Form(""),
    file: UploadFile = File(...),
):
    config = _load_config()
    shard_save_root = config.get("received_shards_dir", "shards/incoming_shards")
    dest = output_dir or f"{shard_save_root}/api/from-rank-{rank}"

    data = await file.read()
    shard = st_load(BytesIO(data))

    try:
        shard_path, metadata_path = save_received_data_shard(
            shard=shard,
            metadata={"rank": rank, "role": role, "host": host},
            output_dir=dest,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"shard_path": shard_path, "metadata_path": metadata_path, "rank": rank}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
