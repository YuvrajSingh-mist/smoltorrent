import argparse
import logging
import subprocess
from pathlib import Path

import httpx
import yaml

from utils.common_utils import model_id_to_dir_name
from utils.check_workers import ping_worker
from utils.log_utils import log_shard_progress

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smoltorrent")

API_BASE = "http://localhost:8000"
CONFIG_PATH = Path(__file__).parent / "configs" / "config.yaml"
# Remote shard root on each worker node (~ expands to the SSH user's home)
REMOTE_SHARDS_ROOT = "~/Desktop/smoltorrent/shards/incoming_shards"

def _load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _count_remote_shards(model_name: str, workers: list[dict]) -> tuple[int, list[dict]]:
    """SSH into each worker using the system ssh binary and count .safetensors shard files.

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


def gather_shards(model_id: str) -> dict:
    resp = httpx.post(
        f"{API_BASE}/gather-shards",
        params={"model_id": model_id},
        timeout=300.0,
    )
    try:
        body = resp.json()
    except Exception:
        resp.raise_for_status()
        raise
    if resp.is_error:
        detail = body.get("detail", {})
        log_shard_progress(logger, detail.get("gathered", []), detail.get("errors", []))
        resp.raise_for_status()
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolTorrent — distributed shard gather")
    parser.add_argument(
        "--model-id",
        metavar="MODEL_ID",
        required=True,
        help="HuggingFace model ID to gather, e.g. mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    )
    args = parser.parse_args()

    config = _load_config()
    workers = config["devices_config"]["workers"]
    num_workers = len(workers)
    model_name = model_id_to_dir_name(args.model_id)

    logger.info("Checking worker heartbeats (%d workers)...", num_workers)
    heartbeat_results = []
    for w in workers:
        host_alias = w.get("host") or w.get("device")
        alive, reason = ping_worker(host_alias, w["ip"], w["port"], w["rank"])
        status = "alive" if alive else f"UNREACHABLE ({reason})"
        logger.info("  rank %d (%s @ %s:%d): %s", w["rank"], host_alias, w["ip"], w["port"], status)
        heartbeat_results.append({"rank": w["rank"], "host": host_alias, "ip": w["ip"], "port": w["port"], "alive": alive})
    dead = [r for r in heartbeat_results if not r["alive"]]
    if dead:
        logger.error(
            "%d/%d workers unreachable — aborting. Dead: %s",
            len(dead), num_workers,
            ", ".join(f"{r['host']}:{r['port']}" for r in dead),
        )
        return

    logger.info("All %d workers alive — checking shards for %s...", num_workers, model_name)
    found, per_worker = _count_remote_shards(model_name, workers)

    for w in per_worker:
        logger.info("  rank %d (%s @ %s): %d shard(s)", w["rank"], w["host"], w["ip"], w["found"])

    if found == 0:
        logger.warning(
            "0/%d shards available for %s — model not found on any worker, skipping transfer",
            num_workers, model_name,
        )
        return

    if found < num_workers:
        logger.warning(
            "%d/%d shards available for %s — shards incomplete, skipping transfer",
            found, num_workers, model_name,
        )
        return

    logger.info("%d/%d shards present — proceeding with gather", found, num_workers)
    result = gather_shards(model_id=args.model_id)
    log_shard_progress(logger, result["gathered"], [])
    logger.info("All shards saved → %s", result.get("save_path"))


if __name__ == "__main__":
    main()
