#!/usr/bin/env python3
"""Check whether all configured workers are alive via heartbeat.

Exit 0 if every worker responds "alive", exit 1 if any fail.
"""
import logging
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

logger = logging.getLogger(__name__)

REMOTE_SHARDS_ROOT = "~/Desktop/smoltorrent/shards/incoming_shards"

import yaml
from networking.send_receive import receive_message, send_message


def count_remote_shards(model_name: str, workers: list[dict]) -> tuple[int, list[dict]]:
    """SSH into each worker and count .safetensors shard files.

    Returns (total_count, per_worker_results) where each entry has
    keys: rank, host, ip, found.
    """
    results = []
    total = 0
    for worker in workers:
        host_alias = worker.get("host")
        ip = worker["ip"]
        rank = worker["rank"]
        remote_dir = f"{REMOTE_SHARDS_ROOT}/{model_name}/worker-{rank}"
        cmd = f"find {remote_dir} -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l"
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host_alias, cmd],
                capture_output=True,
                text=True,
                timeout=15,
            )
            count = int(proc.stdout.strip())
        except Exception as e:
            logger.warning("Could not SSH into %s (%s): %s", host_alias, ip, e)
            count = 0
        results.append({"rank": rank, "host": host_alias, "ip": ip, "found": count})
        total += count
    return total, results


def ping_worker(host: str, ip: str, port: int, rank: int, timeout: float = 5.0) -> tuple[bool, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((ip, port))
        sock.settimeout(None)
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
        status = "\033[32m✓ alive\033[0m" if alive else f"\033[31m✗ dead  ({reason})\033[0m"
        print(f"  rank {rank}  {host} ({ip}:{port})  {status}")
        if not alive:
            all_alive = False

    if not all_alive:
        print("\n\033[31mSome workers are not alive. Aborting.\033[0m")
        sys.exit(1)

    print("\n\033[32mAll workers alive.\033[0m")


if __name__ == "__main__":
    main()
