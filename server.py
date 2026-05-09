from collections import defaultdict
import threading
import socket
from typing import Dict, List, Optional, Union
import mlx.core as mx
import logging
import yaml

from time import time
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

            command, rank, data = message

            if command == "parameter_server_reduce":
                logger.info(f"Storing gradients from worker {rank}")
                with lock:
                    data_received[rank] = data
                    
                    logger.info(
                        f"Now have {len(data_received)} gradient sets"
                    )
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



def main():
    
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
    outbound_worker_sockets = {}
    data_received = defaultdict(dict)

    # Get my worker configuration from allToAllTopology
    workers_list = config["devices_config"]["workers"]
    my_worker_config = config["devices_config"]["master"]
    my_port = my_worker_config["port"]
    worker_rank = my_worker_config["rank"]
    
    # Step 1: Each worker binds to its configured port
    HOST_IP = "0.0.0.0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST_IP, my_port))
    sock.listen()  # Allow multiple connections for worker registration
    logger.info(f"Worker {worker_rank} listening on port {my_port}")

    # Step 2: Connect to next worker in linear topology (if not last worker)
    max_retries = 30
    retry_delay = 2

    for _ in range(NUM_WORKERS - 1):
        # Connect to next worker in the chain
        next_worker = next(w for w in workers_list if w["rank"] != worker_rank)
        next_ip = next_worker["ip"]
        next_port = next_worker["port"]
        del workers_list[
            workers_list.index(next_worker)
        ]  # Remove the next worker from the list to avoid duplicate connections

        logger.info(
            f"Worker {worker_rank} will connect to worker {worker_rank + 1} at {next_ip}:{next_port}"
        )
        time.sleep(worker_rank * 0.5)  # Stagger connections

        for attempt in range(max_retries):
            try:
                next_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                next_sock.connect((next_ip, next_port))
                send_message(next_sock, ("register", worker_rank))

                logger.info(
                    f"Worker {worker_rank} connected to worker {worker_rank + 1} at {next_ip}:{next_port}"
                )
                outbound_worker_sockets[next_worker["rank"]] = (
                    next_sock  # This is important because this has the IP + PORT to which the nodes connected to it listen to which is what we have defined and not send stuff to the port we received through sock.accept()!
                )
                break
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection to worker {worker_rank + 1} refused (attempt {attempt + 1}/{max_retries} at IP: {next_ip}:{next_port}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to connect to worker {worker_rank + 1} after {max_retries} attempts"
                    )
                    raise

    # Step 3: Accept connection from all workers
    accept_workers(
        sock,
        NUM_WORKERS,
        workers=workers,
        data_received=data_received,
        step_event=step_event,
        lock=lock
    )

    
    # Send start signal to all workers
    for worker_socket in outbound_worker_sockets.values():
        send_message(worker_socket, "start_training")

    data_received[worker_rank] = chunked_data[worker_rank]  # Add my own data chunk to the received data for training loop
    
    # Wait for all workers
    while True:
        with lock:
            curr_workers_len = len(data_received)

        logger.info(
            f"Received gradients from {curr_workers_len}/{WORLD_SIZE} participants."
        )
        if curr_workers_len < NUM_WORKERS:
            logger.info(f"Waiting for more gradients...")
            step_event.wait()
            step_event.clear()
        else:
            break
        

if __name__ == "__main__":
    main()
