"""CLI entry-point for SmolTorrent.

Usage:
    python main.py --model-id <hf_model_id> --action [store|gather]

  store  — shard the model and push shards to all configured workers.
  gather — pull shards from workers, merge, and download HuggingFace metadata.
"""
import argparse
import logging

from utils.common_utils import fetch_model_metadata, load_config, model_id_to_dir_name
from utils.log_utils import log_shard_progress
from utils.shard_ops import request_gather_shards, request_store_shards

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smoltorrent")


def main() -> None:
    """Parse CLI args, check worker heartbeats, then run store or gather.

    Exits early (returns without error) if any worker is unreachable or if
    fewer shards than workers are found during a gather pre-check.
    """
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

    if args.action == "store":
        logger.info("Storing shards for %s...", model_name)
        result = request_store_shards(model_id=args.model_id)
        for entry in result.get("sent_to", []):
            logger.info("  ✓ rank %d (%s)", entry["rank"], entry["host"])
        logger.info(
            "Stored %d/%d shards for %s",
            len(result.get("sent_to", [])), result.get("num_shards", num_workers), model_name,
        )
        return

    logger.info("Gathering shards for %s...", model_name)
    result = request_gather_shards(model_id=args.model_id)
    log_shard_progress(logger, result["gathered"], result.get("errors", []))
    if result.get("errors"):
        logger.warning(
            "Partial gather: %d/%d shards retrieved — model is incomplete, skipping inference metadata",
            len(result["gathered"]), num_workers,
        )
        return
    logger.info("All shards saved → %s", result.get("save_path"))
    logger.info("Fetching tokenizer and config from HuggingFace Hub...")
    fetch_model_metadata(args.model_id, config)
    logger.info("received_model/ is ready for inference")


if __name__ == "__main__":
    main()
