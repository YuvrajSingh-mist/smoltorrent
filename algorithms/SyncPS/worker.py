"""SyncPS worker process — listens for shard store/send/heartbeat commands over TCP.

Each worker binds a TCP port, handles incoming connections in daemon threads, and
persists shards to disk under ``shards/worker_{rank}/{rel_path}/shard_N.safetensors``.

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

from networking.send_receive import receive_message, send_message, serve_file, receive_file
from utils.common_utils import (
    compute_checksum,
    handle_json_header,
)

from utils.observability import setup_worker
from utils.prometheus_utils import WorkerMetrics
from discovery import advertise_worker

metrics: Optional["WorkerMetrics"] = None

from utils.log_utils import setup_logging
setup_logging()

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
    logger.debug("[syncps] connection opened from %s", caller)
    try:
        msg = receive_message(conn)
        if msg is None:
            logger.warning("[syncps] Empty message from %s", caller)
            return
        command, *_ = msg if isinstance(msg, tuple) else (msg,)
        logger.debug("[syncps] command=%s from %s", command, caller)
        if command == "heartbeat":
            send_message(conn, "alive")
            logger.info(f"[syncps] Heartbeat ack → {caller}")

        elif command == "checksum_sync":
            _, rank, rel_path = msg
            shard_dir = SHARDS_ROOT / f"worker_{rank}" / rel_path
            # Primary shard file is shard_0.safetensors under the new naming convention.
            shard_path = shard_dir / "shard_0.safetensors"
            if not shard_path.exists():
                send_message(conn, ("checksum_sync_result", "missing", rel_path))
                logger.info(f"[syncps] Checksum missing for rank {rank} at {rel_path}")
                return
            checksum_path = shard_dir / "shard_0.checksum"
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
                if not any((SHARDS_ROOT / f"worker_{rank}" / rp).glob("shard_*.safetensors"))
            ]
            send_message(conn, missing)
            logger.info(
                f"[syncps] Crosscheck: {len(rel_paths) - len(missing)}/{len(rel_paths)} present, {len(missing)} missing"
            )

        elif command == "sync":
            _, rank, extensions = msg
            worker_dir = SHARDS_ROOT / f"worker_{rank}"
            existing = []
            if worker_dir.exists():
                seen = set()
                for shard_file in worker_dir.rglob("shard_*.safetensors"):
                    rel = str(shard_file.parent.relative_to(worker_dir))
                    if rel not in seen:
                        seen.add(rel)
                        existing.append(rel)
            send_message(conn, existing)
            logger.info(f"[syncps] Sync: reported {len(existing)} existing path(s) to {caller}")

        elif command == "send_shard":
            # shard_filename defaults to shard_0.safetensors (primary copy)
            _, rank, rel_path, *_rest = msg
            shard_filename = _rest[0] if _rest else "shard_0.safetensors"
            shard_path = SHARDS_ROOT / f"worker_{rank}" / rel_path / shard_filename
            if not shard_path.exists():
                logger.warning(
                    f"[syncps] No shard on disk for rank {rank} at {shard_path}, cannot serve to {caller}"
                )
                send_message(conn, ("send_shard_missing", rank, rel_path))
                return
            # Ack before streaming so the master can detect missing vs. found
            send_message(conn, ("send_shard_ok", rank, rel_path))
            logger.info(f"[syncps] Serving {shard_filename} for rank {rank}: {shard_path}")
            t0 = time.perf_counter()
            shard_bytes = serve_file(conn, str(shard_path))
            elapsed = time.perf_counter() - t0
            if metrics:
                metrics.bytes_sent.labels(rank=str(rank)).inc(shard_bytes)
                metrics.send_ops.labels(rank=str(rank)).inc()
                metrics.send_duration.labels(rank=str(rank)).observe(elapsed)
            logger.info(f"[syncps] Served {shard_filename} to {caller}")

        elif command == "send_shard_range":
            # Sends only the raw tensor data bytes of a shard (no safetensors header framing).
            # Used by the streaming-merge gather path so the coordinator can write each
            # shard's tensor bytes directly into the correct offset of the pre-allocated
            # merged file without loading any tensors into RAM.
            _, rank, rel_path, *_rest = msg
            shard_filename = _rest[0] if _rest else "shard_0.safetensors"
            shard_path = SHARDS_ROOT / f"worker_{rank}" / rel_path / shard_filename
            if not shard_path.exists():
                logger.warning("[syncps] send_shard_range: no shard for rank %d at %s", rank, shard_path)
                send_message(conn, ("send_shard_range_missing", rank, rel_path))
                return
            _, local_data_offset = handle_json_header(str(shard_path))
            data_length = shard_path.stat().st_size - local_data_offset
            send_message(conn, ("send_shard_range_ok", rank, data_length))
            logger.info("[syncps] send_shard_range rank=%d file=%s data_offset=%d length=%d", rank, shard_filename, local_data_offset, data_length)
            serve_file(conn, str(shard_path), local_data_offset, data_length)
            logger.info("[syncps] send_shard_range done rank=%d file=%s", rank, shard_filename)

        elif command == "store_shard":
            # shard_filename defaults to shard_0.safetensors (round 0 = primary)
            _, rank, received_checksum, rel_path, *_rest = msg
            shard_filename = _rest[0] if _rest else "shard_0.safetensors"
            shard_dir = SHARDS_ROOT / f"worker_{rank}" / rel_path
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / shard_filename
            checksum_path = shard_dir / shard_filename.replace(".safetensors", ".checksum")

            # Receive the mini safetensors header the master built from the original file
            tensor_meta = receive_message(conn)
            if not isinstance(tensor_meta, dict):
                logger.error(f"[syncps] Expected tensor metadata dict from {caller}, got {type(tensor_meta)}")
                send_message(conn, ("store_shard_failed", rank, "missing tensor metadata"))
                return

            t0 = time.perf_counter()
            try:
                # Writes [uint64 hdr_len][JSON header][tensor bytes] — a valid safetensors file
                tensor_data_len, hdr_section_size = receive_file(conn, str(shard_path), st_header=tensor_meta)
                elapsed = time.perf_counter() - t0

                if received_checksum is not None:
                    # Checksum was computed on the raw tensor bytes on the master side,
                    # so verify against just the tensor data section of the written file.
                    actual = compute_checksum(str(shard_path), offset=hdr_section_size, length=tensor_data_len)
                    if actual != received_checksum:
                        logger.error(f"[syncps] Checksum mismatch for rank {rank} ({shard_filename}) from {caller}")
                        shard_path.unlink(missing_ok=True)
                        if metrics:
                            metrics.store_errors.labels(rank=str(rank)).inc()
                        send_message(conn, ("store_shard_failed", rank, "checksum mismatch"))
                        return

                if metrics:
                    metrics.bytes_recv.labels(rank=str(rank)).inc(shard_path.stat().st_size)
                    metrics.store_ops.labels(rank=str(rank)).inc()
                    metrics.store_duration.labels(rank=str(rank)).observe(elapsed)

                # Store checksum of whole file for future integrity checks
                checksum_path.write_text(compute_checksum(shard_path))
                logger.info(f"[syncps] Stored {shard_filename} for rank {rank} from {caller} → {shard_path}")
                send_message(conn, ("store_shard_done", rank, str(shard_path)))
            except Exception as e:
                logger.error("[syncps] Failed to save %s for rank %d: %s", shard_filename, rank, e, exc_info=True)
                if metrics:
                    metrics.store_errors.labels(rank=str(rank)).inc()
                send_message(conn, ("store_shard_failed", rank, str(e)))

        else:
            logger.warning("[syncps] Unknown command '%s' from %s", command, caller)
    except Exception as e:
        logger.error("[syncps] Unhandled error serving %s: %s", caller, e, exc_info=True)
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
    logger.info("[syncps] Starting SmolTorrent worker rank=%d hostname=%s shards_root=%s",
                worker_rank, hostname, SHARDS_ROOT)

    if port is None:
        my_config = next(
            w for w in config["devices_config"]["workers"] if w["rank"] == worker_rank
        )
        port = int(my_config["port"])
    my_port: int = port

    logger.info("[syncps] Worker rank=%d binding on port=%d master=%s:%s",
                worker_rank, my_port, HOST_IP, PORT)

    threading.Thread(
        target=_shard_listener,
        args=(my_port, logger),
        daemon=True,
    ).start()

    logger.info("[syncps] Worker rank=%d ready on port=%d", worker_rank, my_port)

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
