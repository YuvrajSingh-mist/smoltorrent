"""SyncPS worker process — listens for shard store/send/heartbeat commands over TCP.

Each worker binds a TCP port, handles incoming connections in daemon threads, and
persists shards to disk under ``shards/worker_{rank}/{rel_path}/shard.safetensors``.
"""
import threading
import socket
import sys
from pathlib import Path
import logging
import yaml
# Ensure the parent directory is in the path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from networking.send_receive import receive_message, send_message, _network_metrics
from utils.common_utils import compute_checksum, load_tensors, save_received_data_shard, shard_from_bytes, shard_to_bytes
from utils.log_utils import setup_cluster_logging
from utils.network_metrics import log_metrics

# Setup logging (will be replaced by setup_cluster_logging in run_syncps_server)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


NUM_WORKERS = config["num_workers"]
WORLD_SIZE = NUM_WORKERS + 1  # Total participants including server
HOST_IP = config["devices_config"]["master"][0]["ip"]
PORT = config["devices_config"]["master"][0]["port"]
_PROJECT_ROOT = Path(__file__).parents[2]
SHARDS_ROOT = _PROJECT_ROOT / "shards"



def _label_caller(addr: tuple) -> str:
    """Return a human-readable label for an incoming connection address.

    Args:
        addr: ``(ip, port)`` tuple from ``socket.accept()``.

    Returns:
        String like ``"server/master (192.168.1.1)"`` or ``"worker 'pi3' (192.168.1.3)"``.
    """
    caller_ip = addr[0]
    master_ip = config["devices_config"]["master"][0]["ip"]
    if caller_ip == master_ip:
        return f"server/master ({caller_ip})"
    worker_hosts = {
        w["ip"]: w.get("host") or w.get("device", "unknown")
        for w in config["devices_config"]["workers"]
    }
    if caller_ip in worker_hosts:
        return f"worker '{worker_hosts[caller_ip]}' ({caller_ip})"
    return f"unknown caller ({caller_ip}:{addr[1]})"


def _handle_shard_client(
    conn: socket.socket,
    addr: tuple,
    logger: logging.Logger,
) -> None:
    """Handle a single incoming TCP connection, dispatching on the command in the first message.

    Supported commands:
      ``heartbeat``   — reply ``"alive"``.
      ``send_shard``  — read shard from disk and send bytes back.
      ``store_shard`` — receive shard bytes, verify checksum, save to disk.

    Args:
        conn: Accepted client socket.
        addr: ``(ip, port)`` of the remote caller.
        logger: Logger instance for this worker.
    """
    caller = _label_caller(addr)
    try:
        msg = receive_message(conn)
        if msg is None:
            logger.warning(f"Empty message from {caller}")
            return
        command, *_ = msg if isinstance(msg, tuple) else (msg,)
        if command == "heartbeat":
            send_message(conn, "alive")
            logger.info(f"Heartbeat ack → {caller}")

        elif command == "send_shard":
            _, rank, rel_path = msg
            shard_path = SHARDS_ROOT / f"worker_{rank}" / rel_path / "shard.safetensors"
            if not shard_path.exists():
                logger.warning(f"No shard on disk for rank {rank} at {shard_path}, cannot serve to {caller}")
                send_message(conn, None)
                return
            logger.info(f"Loading shard from disk for rank {rank}: {shard_path}")
            shard_bytes = shard_to_bytes(load_tensors(shard_path))
            send_message(conn, shard_bytes)
            log_metrics(_network_metrics.get_metrics(), logger, "serve-shard-to-api")
            logger.info(f"Served shard to {caller}")

        elif command == "store_shard":
            _, rank, shard_bytes, received_checksum, rel_path = msg
            if shard_bytes is None:
                logger.warning(f"No shard data in store_shard from {caller}")
                send_message(conn, ("store_shard_failed", rank, "no shard data"))
                return
            if received_checksum is not None:
                if compute_checksum(shard_bytes) != received_checksum:
                    logger.error(f"Checksum mismatch for rank {rank} from {caller}")
                    send_message(conn, ("store_shard_failed", rank, "checksum mismatch"))
                    return
            shard = shard_from_bytes(shard_bytes)
            shard_dir = SHARDS_ROOT / f"worker_{rank}" / rel_path
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / "shard.safetensors"
            try:
                from safetensors.torch import save_file
                save_file(shard, str(shard_path))
                logger.info(f"Stored shard for rank {rank} from {caller} → {shard_path}")
                send_message(conn, ("store_shard_done", rank, str(shard_path)))
            except Exception as e:
                logger.error(f"Failed to save shard for rank {rank}: {e}")
                send_message(conn, ("store_shard_failed", rank, str(e)))

        else:
            logger.warning(f"Unknown command '{command}' from {caller}")
    except Exception as e:
        logger.error(f"Error serving from {caller}: {e}")
    finally:
        conn.close()


def _shard_listener(port: int, logger: logging.Logger) -> None:
    """Accept connections forever and spawn a daemon thread per client.

    Args:
        port: TCP port to bind on all interfaces.
        logger: Logger instance for this worker.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen()
    logger.info(f"Shard listener ready on port {port}")
    while True:
        try:
            conn, addr = srv.accept()
            logger.info(f"Incoming connection from {_label_caller(addr)}")
            threading.Thread(
                target=_handle_shard_client,
                args=(conn, addr, logger),
                daemon=True,
            ).start()
        except Exception as e:
            logger.error(f"Shard listener error: {e}")
            break


def run_worker(worker_rank: int, hostname: str) -> None:
    """Initialise logging, start the shard listener, and block forever.

    Args:
        worker_rank: Integer rank of this worker (must match a rank in config).
        hostname: Human-readable hostname used in log file naming.
    """
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

    my_config = next(
        w for w in config["devices_config"]["workers"] if w["rank"] == worker_rank
    )
    my_port = my_config["port"]

    threading.Thread(
        target=_shard_listener,
        args=(my_port, logger),
        daemon=True,
    ).start()

    logger.info(f"Worker {worker_rank} ready — listening on port {my_port}")
    threading.Event().wait()  # block forever; shard listener runs as daemon threads

def main() -> None:
    """CLI entry-point. Expects ``<worker_rank>`` and ``<hostname>`` as positional args."""
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python algorithms/SyncPS/worker.py <worker_rank> <hostname>")
    run_worker(int(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
