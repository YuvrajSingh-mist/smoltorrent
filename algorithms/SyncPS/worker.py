"""SyncPS worker process — listens for shard store/send/heartbeat commands over TCP.

Each worker binds a TCP port, handles incoming connections in daemon threads, and
persists shards to disk under ``shards/worker_{rank}/{rel_path}/shard.safetensors``.

The server / master side lives in ``backend/api.py`` (not a separate ``server.py``).
It connects to each worker over TCP and sends ``store_shard`` / ``send_shard`` /
``heartbeat`` / ``checksum_sync`` / ``sync`` / ``all_shards_present`` commands.
"""

import threading
import socket
import sys
import time
from pathlib import Path
from typing import Optional
import logging
import yaml
import argparse

sys.path.insert(0, str(Path(__file__).parents[2]))

from networking.send_receive import receive_message, send_message, network_metrics, serve_file_sendfile
from utils.common_utils import (
    compute_checksum,
    load_tensors,
    shard_from_bytes,
    shard_to_bytes,
)
from utils.network_metrics import log_network_metrics
from utils.observability import setup_worker
from utils.prometheus_utils import WorkerMetrics
from discovery import advertise_worker

metrics: Optional["WorkerMetrics"] = None

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
            logger.warning(f"[syncps] Empty message from {caller}")
            return
        command, *_ = msg if isinstance(msg, tuple) else (msg,)
        if command == "heartbeat":
            send_message(conn, "alive")
            logger.info(f"[syncps] Heartbeat ack → {caller}")

        elif command == "checksum_sync":
            _, rank, rel_path = msg
            shard_path = SHARDS_ROOT / f"worker_{rank}" / rel_path / "shard.safetensors"
            checksum_path = shard_path.parent / "shard.checksum"
            if not checksum_path.exists():
                cksum = compute_checksum(shard_path)
                checksum_path.write_text(cksum)
                logger.info(f"[syncps] Bootstrapped checksum for rank {rank} at {rel_path}")
                send_message(conn, ("checksum_sync_result", "ok", rel_path))
            else:
                current = compute_checksum(shard_path)
                stored = checksum_path.read_text().strip()
                status = "ok" if current == stored else "mismatch"
                send_message(conn, ("checksum_sync_result", status, rel_path))
                logger.info(f"[syncps] Checksum {status} for rank {rank} at {rel_path}")

        elif command == "all_shards_present":
            _, rank, rel_paths = msg
            missing = [
                rp
                for rp in rel_paths
                if not (
                    SHARDS_ROOT / f"worker_{rank}" / rp / "shard.safetensors"
                ).exists()
            ]
            send_message(conn, missing)
            logger.info(
                f"[syncps] Crosscheck: {len(rel_paths) - len(missing)}/{len(rel_paths)} present, {len(missing)} missing"
            )

        elif command == "sync":
            _, rank, extensions = (
                msg  # extensions unused: shards are always stored as shard.safetensors
            )
            worker_dir = SHARDS_ROOT / f"worker_{rank}"
            existing = []
            if worker_dir.exists():
                for shard_file in worker_dir.rglob("shard.safetensors"):
                    existing.append(str(shard_file.parent.relative_to(worker_dir)))
            send_message(conn, existing)
            logger.info(f"[syncps] Sync: reported {len(existing)} existing path(s) to {caller}")

        elif command == "send_shard":
            _, rank, rel_path = msg
            shard_path = SHARDS_ROOT / f"worker_{rank}" / rel_path / "shard.safetensors"
            if not shard_path.exists():
                logger.warning(
                    f"[syncps] No shard on disk for rank {rank} at {shard_path}, cannot serve to {caller}"
                )
                send_message(conn, None)
                return
            logger.info(f"[syncps] Loading shard from disk for rank {rank}: {shard_path}")
            t0 = time.perf_counter()
            # shard_bytes = shard_to_bytes(load_tensors(shard_path))
            # send_message(conn, shard_bytes)
            shard_bytes = serve_file_sendfile(conn, str(shard_path))
            elapsed = time.perf_counter() - t0
            if metrics:
                metrics.bytes_sent.labels(rank=str(rank)).inc(shard_bytes)
                metrics.send_ops.labels(rank=str(rank)).inc()
                metrics.send_duration.labels(rank=str(rank)).observe(elapsed)
            log_network_metrics(network_metrics.get_metrics(), logger, "serve-shard-to-api")
            logger.info(f"[syncps] Served shard to {caller}")

        elif command == "store_shard":
            _, rank, shard_bytes, received_checksum, rel_path = msg
            if shard_bytes is None:
                logger.warning(f"[syncps] No shard data in store_shard from {caller}")
                send_message(conn, ("store_shard_failed", rank, "no shard data"))
                return
            if received_checksum is not None:
                if compute_checksum(shard_bytes) != received_checksum:
                    logger.error(f"[syncps] Checksum mismatch for rank {rank} from {caller}")
                    if metrics:
                        metrics.store_errors.labels(rank=str(rank)).inc()
                    send_message(
                        conn, ("store_shard_failed", rank, "checksum mismatch")
                    )
                    return
            if metrics:
                metrics.bytes_recv.labels(rank=str(rank)).inc(len(shard_bytes))
            shard = shard_from_bytes(shard_bytes)
            shard_dir = SHARDS_ROOT / f"worker_{rank}" / rel_path
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / "shard.safetensors"
            t0 = time.perf_counter()
            try:
                from safetensors.torch import save_file

                save_file(shard, str(shard_path))
                elapsed = time.perf_counter() - t0
                cksum = compute_checksum(shard_path)
                (shard_dir / "shard.checksum").write_text(cksum)
                if metrics:
                    metrics.store_ops.labels(rank=str(rank)).inc()
                    metrics.store_duration.labels(rank=str(rank)).observe(elapsed)
                log_network_metrics(
                    network_metrics.get_metrics(), logger, f"store-shard-rank{rank}"
                )
                logger.info(
                    f"[syncps] Stored shard for rank {rank} from {caller} → {shard_path}"
                )
                send_message(conn, ("store_shard_done", rank, str(shard_path)))
            except Exception as e:
                logger.error(f"[syncps] Failed to save shard for rank {rank}: {e}")
                if metrics:
                    metrics.store_errors.labels(rank=str(rank)).inc()
                send_message(conn, ("store_shard_failed", rank, str(e)))

        else:
            logger.warning(f"[syncps] Unknown command '{command}' from {caller}")
    except Exception as e:
        logger.error(f"[syncps] Error serving from {caller}: {e}")
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
    logger.info(f"[syncps] Shard listener ready on port {port}")
    while True:
        try:
            conn, addr = srv.accept()
            logger.info(f"[syncps] Incoming connection from {_label_caller(addr)}")
            threading.Thread(
                target=_handle_shard_client,
                args=(conn, addr, logger),
                daemon=True,
            ).start()
        except Exception as e:
            logger.error(f"[syncps] Shard listener error: {e}")
            break


def run_worker(worker_rank: int, hostname: str, port: Optional[int] = None) -> None:
    """Initialise logging, start the shard listener, and block forever.

    Args:
        worker_rank: Integer rank of this worker.
        hostname: Human-readable hostname used in log file naming.
        port: TCP port to bind. If not given, looked up from config by rank.
    """
    logger = logging.getLogger(f"[WORKER-{worker_rank}]")

    global metrics
    metrics = setup_worker(
        rank=worker_rank,
        hostname=hostname,
        log_dir=config.get("log_dir"),
    )
    logger.info("[syncps] Starting SmolTorrent...")

    if port is None:
        my_config = next(
            w for w in config["devices_config"]["workers"] if w["rank"] == worker_rank
        )
        port = int(my_config["port"])
    my_port: int = port

    threading.Thread(
        target=_shard_listener,
        args=(my_port, logger),
        daemon=True,
    ).start()

    logger.info(f"[syncps] Worker {worker_rank} ready — listening on port {my_port}")

    # Advertise this worker over mDNS so the master can discover it automatically.
    # Runs for the lifetime of the process; close() is called on normal exit.
    advertiser = advertise_worker(rank=worker_rank, port=my_port, hostname=hostname)
    logger.info(
        f"[syncps] Worker {worker_rank} advertising on mDNS as smoltorrent-rank-{worker_rank}"
    )

    try:
        threading.Event().wait()  # block forever; shard listener runs as daemon threads
    finally:
        advertiser.close()


def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int)
    parser.add_argument("hostname")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    run_worker(args.rank, args.hostname, port=args.port)


if __name__ == "__main__":
    main()
