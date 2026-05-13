import argparse
import logging

from utils.check_workers import count_remote_shards, ping_worker
from utils.common_utils import fetch_model_metadata, load_config, model_id_to_dir_name
from utils.log_utils import log_shard_progress
from utils.shard_ops import request_gather_shards, request_store_shards

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smoltorrent")


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolTorrent — distributed shard store/gather")
    parser.add_argument(
        "--model-id",
        metavar="MODEL_ID",
        required=True,
        help="HuggingFace model ID, e.g. mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    )
    parser.add_argument(
        "--action",
        choices=["store", "gather"],
        default="gather",
        help="store: shard model and push to workers. gather: pull shards back and merge (default: gather)",
    )
    args = parser.parse_args()

    config = load_config()
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

    if args.action == "store":
        logger.info("All %d workers alive — storing shards for %s...", num_workers, model_name)
        result = request_store_shards(model_id=args.model_id)
        for entry in result.get("sent_to", []):
            logger.info("  ✓ rank %d (%s)", entry["rank"], entry["host"])
        logger.info(
            "Stored %d/%d shards for %s",
            len(result.get("sent_to", [])), result.get("num_shards", num_workers), model_name,
        )
        return

    logger.info("All %d workers alive — checking shards for %s...", num_workers, model_name)
    found, per_worker = count_remote_shards(model_name, workers)

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
    result = request_gather_shards(model_id=args.model_id)
    log_shard_progress(logger, result["gathered"], [])
    logger.info("All shards saved → %s", result.get("save_path"))
    logger.info("Fetching tokenizer and config from HuggingFace Hub...")
    fetch_model_metadata(args.model_id, config)
    logger.info("received_model/ is ready for inference")


if __name__ == "__main__":
    main()
