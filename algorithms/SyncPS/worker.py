from collections import defaultdict
import threading
import socket
from typing import Dict, List, Optional, Union
import mlx.core as mx
import logging
import sys
import yaml
import subprocess
from time import time


from networking.send_receive import receive_message, send_message
from utils.common_utils import chunk_data, save_received_data_shard
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
HOST_IP = config["devices_config"]["master"]["ip"]
PORT = config["devices_config"]["master"]["port"]
SHARD_SAVE_ROOT = config.get("received_shards_dir", "shards/incoming_shards")

def load_data(file_path: str) -> dict:
    """Load data from a file."""
    data = mx.load(file_path)
    return data


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

            command, recv_step, rank, grads = message

            if command == "all_gather":
                logger.info(
                    f"Received message '{command}' from worker {addr} (rank {rank}) for step {recv_step}"
                )
                logger.info(f"[Step {recv_step}] Storing gradients from worker {rank}")

                with lock:
                    data_received[recv_step][rank] = grads
                    logger.info(
                        f"[Step {recv_step}] Now have {len(data_received[recv_step])} gradient sets"
                    )

                # reduced_grads = reduce(data_received[recv_step], len(data_received[recv_step]))
                step_event.set()

            elif command == "down":
                logger.info(f"Worker {addr} requested shutdown")
                break

        except Exception as e:
            logger.error(f"Error handling worker {addr}: {e}")
            break
    logger.info(f"Worker {addr} disconnected")
    conn.close()


def connect_to_server(
    host: str, port: int, max_retries: int = 60, retry_delay: float = 3.0
) -> socket.socket:
    """Connect to server with retry logic."""
    # Ping to warm up ARP cache (especially important for WiFi networks)
    logger.info(f"Warming up ARP cache by pinging {host}...")
    try:
        subprocess.run(
            ["ping", "-c", "3", "-W", "1000", host], capture_output=True, timeout=10
        )
    except Exception as e:
        logger.warning(f"ARP warmup ping failed: {e}")

    for attempt in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 10 second timeout for connection
        try:
            sock.connect((host, port))
            sock.settimeout(None)  # Remove timeout after connection
            logger.info(
                f"Connected to server at {host}:{port} on attempt {attempt + 1}"
            )
            return sock
        except (OSError, ConnectionRefusedError, socket.timeout) as e:
            sock.close()  # Close the failed socket
            # Re-ping every 5 attempts to keep ARP fresh
            if attempt > 0 and attempt % 5 == 0:
                logger.info(f"Re-pinging {host} to refresh ARP cache...")
                try:
                    subprocess.run(
                        ["ping", "-c", "2", "-W", "1000", host],
                        capture_output=True,
                        timeout=5,
                    )
                except Exception:
                    pass
            if attempt < max_retries - 1:
                logger.warning(
                    f"Connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to connect to server after {max_retries} attempts"
                )
                raise
    # This should never be reached, but just in case
    raise RuntimeError("Failed to connect to server")



def accept_workers(
    sock: socket.socket,
    NUM_WORKERS: int,
    workers: dict,
    step_event: threading.Event,
    lock: threading.Lock,
    data_received: Optional[Union[Dict, List]] = None
) -> None:
    # Accept connections and wait for registration
    expected_peers = max(NUM_WORKERS - 1, 0)
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
                        workers,
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

    logger.info("All workers connected. Starting training...")



def run_worker(worker_rank: int):
     
    global logger

    # Configure centralized logging
    setup_cluster_logging(
        logger=logger,
        component="server",
        rank=None,
        hostname=config["devices_config"]["master"]["ip"],
        log_dir=config.get("log_dir", "/tmp/smolcluster-logs"),
        algorithm="syncps",
    )
    logger.info("Starting SmolTorrent...")
    
     # Thread-safe data structures
    step_event = threading.Event()
    lock = threading.Lock()
    workers = {}
    outbound_worker_sockets = {}
    data_received = defaultdict(dict)
 
     # Connect to server
    sock = connect_to_server(HOST_IP, PORT)

    # Register with server
    logger.info(f"Registering as worker {worker_rank} with server...")
    send_message(sock, ("register", worker_rank))

    logger.info(
        f"Worker {worker_rank} connected to server at {HOST_IP}:{PORT}"
    )

    # Wait for start signal
    logger.info("Waiting for start_training signal from server...")
    while True:
        recv_command = receive_message(sock)
        if recv_command == "start_training":
            logger.info("Received start_training command from server.")
            break
    
    send_message(sock, ("parameter_server_reduce", worker_rank, data_received[worker_rank]))

    # Receive updated weights from server
    logger.info("Waiting for model weights from server")
    data_recv = receive_message(sock)
    command, recv_step, shard = data_recv
    logger.info(
        f"Received '{command}' from server for step {recv_step}"
    )

    save_received_data_shard(
        shard=shard,
        metadata={
            "role": "worker_received",
            "rank": worker_rank,
            "step": recv_step,
            "command": command,
            "source_host": HOST_IP,
            "source_port": PORT,
        },
        output_dir=f"{SHARD_SAVE_ROOT}/worker-{worker_rank}",
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python algorithms/SyncPS/worker.py <worker_rank>")
    run_worker(int(sys.argv[1]))


if __name__ == "__main__":
    main()
