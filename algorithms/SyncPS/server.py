from collections import defaultdict
import httpx
from io import BytesIO
import threading
import socket
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import yaml
import time
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save as st_save

# Ensure the parent directory is in the path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from networking.send_receive import receive_message, send_message
from utils.common_utils import chunk_data
from utils.log_utils import setup_cluster_logging

# Setup logging (will be replaced by setup_cluster_logging in run_syncps_server)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[SERVER]")

with open("configs/config.yaml", "r") as f:
    config = yaml. safe_load(f)


NUM_WORKERS = config["num_workers"]
WORLD_SIZE = NUM_WORKERS + 1  # Total participants including server
SHARD_SAVE_ROOT = config.get("received_shards_dir", "shards/incoming_shards")


def load_data(file_path: str) -> dict:
    """Load data from a safetensors file."""
    return safetensors_load_file(file_path)


def handle_worker(
    conn: socket.socket,
    addr: tuple[str, int],
    data_received: Optional[Union[Dict, List]],
    step_event: threading.Event,
    lock: threading.Lock,
) -> None:
    """Handle individual worker connections and gradient reception."""
    logger.info(f"Handling worker at {addr}")

    while True:
        try:
            message = receive_message(conn)

            # Handle connection closed or empty message
            if message is None:
                # logger.info(f"Worker {addr} closed connection")
                logger.warning(f"Received empty message from worker {addr}")
                break

            logger.debug(len(message))

            command, rank, data = message

            if command == "store_shard":
                logger.info(f"Storing shard from worker {rank}")
                with lock:
                    data_received[rank] = data

                    buf = BytesIO()
                    st_save(buf, data)
                    buf.seek(0)
                    resp = httpx.post(
                        "http://localhost:8000/store-shard",
                        data={
                            "rank": rank,
                            "role": "server_received",
                            "host": str(addr[0]),
                            "output_dir": f"{SHARD_SAVE_ROOT}/server/from-rank-{rank}",
                        },
                        files={"file": ("shard.safetensors", buf, "application/octet-stream")},
                        timeout=60.0,
                    )
                    if resp.is_success:
                        logger.info("Stored shard from worker %d → %s", rank, resp.json()["shard_path"])
                    else:
                        logger.error("store-shard API failed for rank %d: %s", rank, resp.text)

                    logger.info("Now have %d shard sets", len(data_received))
                step_event.set()

            elif command == "down":
                logger.info(f"Worker {addr} requested shutdown")
                break

        except Exception as e:
            logger.error(f"Error handling worker {addr}: {e}")
            break
    logger.info(f"Worker {addr} disconnected")
    conn.close()


def accept_workers(
    sock: socket.socket,
    NUM_WORKERS: int,
    workers: dict,
    step_event: threading.Event,
    lock: threading.Lock,
    data_received: Optional[Union[Dict, List]] = None
) -> None:
    # Accept connections and wait for registration
    expected_peers = max(NUM_WORKERS, 0)
    registered_workers = {}  # rank -> socket
    while len(registered_workers) < expected_peers:
        client_socket, client_address = sock.accept()
        logger.info(f"Accepted connection from {client_address}")

        # Wait for registration message
        try:
            message = receive_message(client_socket)
            if message is None:
                logger.warning(
                    f"Connection from {client_address} closed before registration"
                )
                client_socket.close()
                break

            command, worker_rank = message
            if command == "register":
                logger.info(f"Worker {worker_rank} registered from {client_address}")
                registered_workers[worker_rank] = client_socket
                workers[client_address] = client_socket
                threading.Thread(
                    target=handle_worker,
                    args=(
                        client_socket,
                        client_address,
                        data_received,
                        step_event,
                        lock,
                    ),
                    daemon=True,
                ).start()
            else:
                logger.warning(f"Unexpected message from {client_address}: {command}")
                client_socket.close()
                break

        except Exception as e:
            logger.error(f"Error during registration from {client_address}: {e}")
            client_socket.close()
            break

    logger.info("All workers connected.")
    return registered_workers


def run_server():
    
    # Configure centralized logging
    setup_cluster_logging(
        logger=logger,
        component="server",
        rank=None,
        hostname=config["devices_config"]["master"][0]["host"],
        log_dir=config.get("log_dir", "/tmp/smolcluster-logs"),
        algorithm="syncps",
    )
    logger.info("Starting SmolTorrent...")
    
    logger.info("Loading data...")
    data = load_data(config["data_path"])
    total_bytes = sum(v.nbytes for v in data.values())
    logger.info(f"Loaded {len(data)} tensors, total size {total_bytes / 1024**2:.1f} MB")
    
    chunked_data = chunk_data(data, n_chunks=NUM_WORKERS)  # 10 chunks
    logger.info(f"Split data into {len(chunked_data)} chunks")
    
     # Thread-safe data structures
    step_event = threading.Event()
    lock = threading.Lock()
    workers = {}
    data_received = defaultdict(dict)

    # Get my worker configuration from allToAllTopology
    workers_list = config["devices_config"]["workers"]
    my_worker_config = config["devices_config"]["master"][0]
    my_port = my_worker_config["port"]
    worker_rank = my_worker_config["rank"]
    
    # Step 1: Each worker binds to its configured port
    HOST_IP = "0.0.0.0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST_IP, my_port))
    sock.listen()  # Allow multiple connections for worker registration
    logger.info(f"Worker {worker_rank} listening on port {my_port}")

    # Step 3: Accept connection from all workers
    registered_workers = accept_workers(
        sock,
        NUM_WORKERS,
        workers=workers,
        data_received=data_received,
        step_event=step_event,
        lock=lock
    )

    logger.info(f"Registered workers: {list(registered_workers.keys())}")
    logger.info("Sending start signal to all workers...")
    # Send start signal to all workers
    for worker_socket in registered_workers.values():
        send_message(worker_socket, "start")
    logger.info("Start signal sent to all workers.")
    
  
    logger.info("Sending data shards to workers...")
    for rank, worker_socket in registered_workers.items():
        send_message(worker_socket, ("store_shard", rank, chunked_data[rank - 1])) # -1 for 0-indexing of chunked_data because of rank 0 being assigned to master which has no role to play here

    logger.info("Data shards sent to all workers.")


if __name__ == "__main__":
    run_server()
