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
SHARDS_ROOT = _PROJECT_ROOT / "shards" / "incoming_shards"
_MODEL_NAME = Path(config.get("data_path", "model")).parent.name



def _label_caller(addr: tuple) -> str:
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
            rank = msg[1] if len(msg) > 1 else None
            shard_dir = SHARDS_ROOT / _MODEL_NAME / f"worker-{rank}"
            existing = sorted(shard_dir.glob("*.safetensors")) if shard_dir.exists() else []
            if not existing:
                logger.warning(f"No shard on disk for rank {rank} at {shard_dir}, cannot serve to {caller}")
                return
            logger.info(f"Loading shard from disk for rank {rank}: {existing[0]}")
            shard_bytes = shard_to_bytes(load_tensors(existing[0]))
            send_message(conn, shard_bytes)
            log_metrics(_network_metrics.get_metrics(), logger, "serve-shard-to-api")
            logger.info(f"Served shard to {caller}")

        elif command == "store_shard":
            _, rank, shard_bytes, received_checksum = msg
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
            shard_path, metadata_path, ok, err = save_received_data_shard(
                shard=shard,
                metadata={"role": "worker_received", "rank": rank, "source_host": caller},
                output_dir=SHARDS_ROOT / _MODEL_NAME / f"worker-{rank}",
            )
            if ok:
                logger.info(f"Stored shard for rank {rank} from {caller} → {shard_path}")
                send_message(conn, ("store_shard_done", rank, shard_path, metadata_path))
            else:
                logger.error(f"Failed to save shard for rank {rank}: {err}")
                send_message(conn, ("store_shard_failed", rank, err))

        else:
            logger.warning(f"Unknown command '{command}' from {caller}")
    except Exception as e:
        logger.error(f"Error serving shard to {caller}: {e}")
    finally:
        conn.close()


def _shard_listener(port: int, logger: logging.Logger) -> None:
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
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python algorithms/SyncPS/worker.py <worker_rank> <hostname>")
    run_worker(int(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
