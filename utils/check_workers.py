#!/usr/bin/env python3
"""Check whether all configured workers are alive via heartbeat.

Exit 0 if every worker responds "alive", exit 1 if any fail.
"""

import logging
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from networking.send_receive import receive_message, send_message

logger = logging.getLogger(__name__)

REMOTE_SHARDS_ROOT = "~/Desktop/smoltorrent/shards"


def count_remote_shards(
    model_name: str,
    workers: list[dict],
    extensions: Optional[List[str]] = None,
) -> Tuple[int, list]:
    """SSH into each worker and count shard files matching ``extensions``.

    Args:
        model_name: Model subdirectory under each worker's shards dir.
        workers: List of worker config dicts from config.yaml.
        extensions: File extensions to match. Defaults to ``['.safetensors']``.

    Returns:
        Tuple of (workers_with_any_shards, per_worker_results). Each result has
        keys: ``rank``, ``host``, ``ip``, ``remote_dir``, ``found``.
    """
    if extensions is None:
        extensions = [".safetensors"]

    name_clauses = " -o ".join(f"-name '*{ext}'" for ext in extensions)
    find_expr = (
        f"\\( {name_clauses} \\)" if len(extensions) > 1 else f"-name '*{extensions[0]}'"
    )

    results = []
    workers_with_shards = 0
    for worker in workers:
        host_alias = worker.get("host")
        ip = worker["ip"]
        rank = worker["rank"]
        remote_dir = f"{REMOTE_SHARDS_ROOT}/worker_{rank}/{model_name}"
        cmd = f"find {remote_dir} {find_expr} 2>/dev/null | wc -l"
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", str(host_alias or ip), cmd],
                capture_output=True,
                text=True,
                timeout=15,
            )
            count = int(proc.stdout.strip() or "0")
        except Exception as e:
            logger.warning("[check] Could not SSH into %s (%s): %s", host_alias, ip, e)
            count = 0
        results.append({"rank": rank, "host": host_alias, "ip": ip, "remote_dir": remote_dir, "found": count})
        if count > 0:
            workers_with_shards += 1
    return workers_with_shards, results


def ping_worker(
    host: str, ip: str, port: int, rank: int, timeout: float = 0.5
) -> tuple[bool, str]:
    """Send a heartbeat to a worker and check for an ``"alive"`` response.

    Args:
        host: Human-readable host alias (used only for logging).
        ip: Worker IP address.
        port: Worker TCP port.
        rank: Worker rank (used only for logging).
        timeout: Socket connect/receive timeout in seconds.

    Returns:
        Tuple of (alive, reason_string).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((ip, port))
        send_message(sock, ("heartbeat",))
        response = receive_message(sock)
        if response == "alive":
            return True, "alive"
        return False, f"unexpected response: {response!r}"
    except (OSError, ConnectionRefusedError, TimeoutError, socket.timeout) as e:
        return False, str(e)
    finally:
        sock.close()


def main() -> None:
    """CLI entry-point: ping every configured worker and print a status table. Exits 1 if any are dead."""
    config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
    with config_path.open() as f:
        config = yaml.safe_load(f)

    workers = config["devices_config"]["workers"]
    all_alive = True

    for w in workers:
        host = w.get("host") or w.get("device", "unknown")
        ip = w["ip"]
        port = w["port"]
        rank = w["rank"]

        alive, reason = ping_worker(host, ip, port, rank)
        status = (
            "\033[32m✓ alive\033[0m" if alive else f"\033[31m✗ dead  ({reason})\033[0m"
        )
        print(f"  rank {rank}  {host} ({ip}:{port})  {status}")
        if not alive:
            all_alive = False

    if not all_alive:
        print("\n\033[31mSome workers are not alive. Aborting.\033[0m")
        sys.exit(1)

    print("\n\033[32mAll workers alive.\033[0m")


if __name__ == "__main__":
    main()
