#!/usr/bin/env python3
"""Check whether all configured workers are alive via heartbeat.

Exit 0 if every worker responds "alive", exit 1 if any fail.
"""
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import yaml
from networking.send_receive import receive_message, send_message


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
