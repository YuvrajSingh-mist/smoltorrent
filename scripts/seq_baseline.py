"""Measure gather wall-clock time using the old sequential pattern.

The old gather (commit e4e04fc) pulled shards one worker at a time in a for-loop.
This script replicates that exact behaviour so we have a real measured baseline
to compare against the current parallel ThreadPoolExecutor gather.

Usage:
    uv run python scripts/seq_baseline.py <rel_path>

Example:
    uv run python scripts/seq_baseline.py Qwen2.5-0.5B-instruct-bf16/gaming/latest
    uv run python scripts/seq_baseline.py LFM2.5-350M-bf16/test-watcher/latest
"""

import sys
import time
import socket
import yaml
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from networking.send_receive import receive_message, send_message  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "config.yaml"


def _connect(
    ip: str, port: int, rank: int, retries: int = 3, delay: float = 2.0
) -> socket.socket:
    for attempt in range(1, retries + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((ip, port))
            sock.settimeout(None)
            return sock
        except OSError as e:
            print(f"  rank {rank}: connect attempt {attempt} failed — {e}")
            if attempt < retries:
                time.sleep(delay)
    raise ConnectionError(f"rank {rank} unreachable after {retries} attempts")


def sequential_gather(rel_path: str):
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    workers = cfg["devices_config"]["workers"]

    print(f"\n=== SEQUENTIAL GATHER — {rel_path} ===")
    t0 = time.monotonic()

    shards = []
    for worker in workers:
        rank = worker["rank"]
        host = worker.get("host") or worker.get("device")
        wt = time.monotonic()
        sock = _connect(worker["ip"], worker["port"], rank)
        send_message(sock, ("send_shard", rank, rel_path))
        shard_bytes = receive_message(sock)
        sock.close()
        elapsed = time.monotonic() - wt
        print(f"  rank {rank} ({host}): {elapsed:.1f}s")
        if shard_bytes is None:
            raise RuntimeError(f"rank {rank} returned no shard")
        shards.append(shard_bytes)

    wall = time.monotonic() - t0
    print(f"Wall: {wall * 1000:.0f} ms  ({wall:.1f} s)")
    return wall


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rel_path = sys.argv[1]
    sequential_gather(rel_path)
