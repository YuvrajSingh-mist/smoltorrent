from collections import defaultdict
import threading
import socket
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import yaml
import subprocess
import time
# Ensure the parent directory is in the path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from networking.send_receive import receive_message, send_message
from utils.common_utils import chunk_data, save_received_data_shard
from utils.log_utils import setup_cluster_logging

# Setup logging (will be replaced by setup_cluster_logging in run_syncps_server)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

with open("configs/config.yaml", "r") as f:
    config = yaml. safe_load(f)


NUM_WORKERS = config["num_workers"]
WORLD_SIZE = NUM_WORKERS + 1  # Total participants including server
HOST_IP = config["devices_config"]["master"][0]["ip"]
PORT = config["devices_config"]["master"][0]["port"]
SHARD_SAVE_ROOT = config.get("received_shards_dir", "shards/incoming_shards")



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



def _label_caller(addr: tuple) -> str:
    caller_ip = addr[0]
    master_ip = config["devices_config"]["master"][0]["ip"]
    if caller_ip == master_ip:
        return f"server/master ({caller_ip})"
    worker_hosts = {w["ip"]: w.get("host") or w.get("device", "unknown") for w in config["devices_config"]["workers"]}
    if caller_ip in worker_hosts:
        return f"worker '{worker_hosts[caller_ip]}' ({caller_ip})"
    return f"unknown caller ({caller_ip}:{addr[1]})"


def _handle_shard_client(
    conn: socket.socket,
    addr: tuple,
    shard_container: list,
    shard_ready: threading.Event,
) -> None:
    caller = _label_caller(addr)
    try:
        shard_ready.wait(timeout=120)
        msg = receive_message(conn)
        if msg is None:
            logger.warning(f"Empty message from {caller}")
            return
        command, _rank = msg
        if command == "send_shard":
            send_message(conn, shard_container[0])
            logger.info(f"Served shard to {caller}")
            
        elif command == 'store_shard':
            logger.info(f"Storing received shard for rank {worker_rank}")
            save_received_data_shard(
                shard=shard,
                metadata={
                    "role": "worker_received",
                    "rank": worker_rank,
                    "command": command,
                    "source_host": HOST_IP,
                    "source_port": PORT,
                },
                output_dir=f"{SHARD_SAVE_ROOT}/worker-{worker_rank}",
            )
        else:
            logger.warning(f"Unknown command '{command}' from {caller}")
    except Exception as e:
        logger.error(f"Error serving shard to {caller}: {e}")
    finally:
        conn.close()


def _shard_listener(
    port: int,
    shard_container: list,
    shard_ready: threading.Event,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen()
    logger.info(f"Shard listener ready on port {port}")
    while True:
        try:
            conn, addr = sock.accept()
            logger.info(f"Incoming connection from {_label_caller(addr)}")
            threading.Thread(
                target=_handle_shard_client,
                args=(conn, addr, shard_container, shard_ready),
                daemon=True,
            ).start()
        except Exception as e:
            logger.error(f"Shard listener error: {e}")
            break


def run_worker(worker_rank: int, hostname: str):

    logger = logging.getLogger(f"[WORKER-{worker_rank}]")

    # Configure centralized logging
    setup_cluster_logging(
        logger=logger,
        component="worker",
        rank=worker_rank,
        hostname=hostname,
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

    # Start shard listener so the API can pull shards from this worker
    my_config = next(
        w for w in config["devices_config"]["workers"] if w["rank"] == worker_rank
    )
    my_port = my_config["port"]
    shard_container: list = []
    shard_ready = threading.Event()
    threading.Thread(
        target=_shard_listener,
        args=(my_port, shard_container, shard_ready),
        daemon=True,
    ).start()

     # Connect to server
    sock = connect_to_server(HOST_IP, PORT)

    # Register with server
    logger.info(f"Registering as worker {worker_rank} with server...")
    send_message(sock, ("register", worker_rank))

    logger.info(
        f"Worker {worker_rank} connected to server at {HOST_IP}:{PORT}"
    )

    # Wait for start signal
    logger.info("Waiting for start signal from server...")
    while True:
        recv_command = receive_message(sock)
        if recv_command == "start":
            logger.info("Received start command from server.")
            break

    # Receive our assigned shard from the server
    logger.info("Waiting for shard assignment from server...")
    command, rank, shard = receive_message(sock)
    logger.info(f"Received '{command}' for rank {rank} from server")

    # Make the shard available to the listener thread
    shard_container.append(shard)
    shard_ready.set()

    while True:
        if command == 'store_shard':
            logger.info(f"Storing received shard for rank {worker_rank}")
            save_received_data_shard(
                shard=shard,
                metadata={
                    "role": "worker_received",
                    "rank": worker_rank,
                    "command": command,
                    "source_host": HOST_IP,
                    "source_port": PORT,
                },
                output_dir=f"{SHARD_SAVE_ROOT}/worker-{worker_rank}",
            )
        elif command == 'send_shard':
            logger.info(f"Received shard to send back to server for rank {worker_rank}")
            send_message(sock, ("send_shard", worker_rank, shard))

        command = receive_message(sock)
         
def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python algorithms/SyncPS/worker.py <worker_rank> <hostname>")
    run_worker(int(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
