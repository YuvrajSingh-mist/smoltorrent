"""CLI entry-point for SmolTorrent.

Usage:
    python main.py store --ckpt-path <absolute_path_to_checkpoint.safetensors>
    python main.py gather --ckpt-rel-path <relative_path, e.g. grpo/run1/step_100>

  store  — shard the checkpoint and push shards to all configured workers.
  gather — pull shards from workers and merge into a single .safetensors file.
"""
import argparse
import logging

from utils.common_utils import fetch_model_metadata, load_config
from utils.shard_ops import request_gather_shards, request_store_shards

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smoltorrent")


def main() -> None:
    """Parse CLI args then run store or gather."""
    parser = argparse.ArgumentParser(description="SmolTorrent — distributed shard store/gather")
    sub = parser.add_subparsers(dest="action", required=True)

    store_p = sub.add_parser("store", help="Shard a checkpoint and push to workers")
    store_p.add_argument(
        "--ckpt-path",
        required=True,
        metavar="PATH",
        help="Absolute path to the checkpoint .safetensors file",
    )

    gather_p = sub.add_parser("gather", help="Pull shards from workers and merge")
    gather_p.add_argument(
        "--ckpt-path",
        required=True,
        metavar="PATH",
        help="Absolute path to the checkpoint file (same path used for store)",
    )
    gather_p.add_argument(
        "--model-id",
        metavar="MODEL_ID",
        default=None,
        help="HuggingFace model ID to fetch tokenizer/config after merge, e.g. mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    )

    args = parser.parse_args()

    if args.action == "store":
        logger.info("Storing shards for %s...", args.ckpt_path)
        request_store_shards(ckpt_path=args.ckpt_path, log_fn=logger.info)
    else:
        logger.info("Gathering shards for %s...", args.ckpt_path)
        request_gather_shards(ckpt_path=args.ckpt_path, log_fn=logger.info)
        if args.model_id:
            logger.info("Fetching tokenizer and config from HuggingFace Hub...")
            config = load_config()
            fetch_model_metadata(args.model_id, config)
            logger.info("Model directory ready for inference")
        else:
            logger.info("Done — merged.safetensors is ready")


if __name__ == "__main__":
    main()
