import logging
import httpx

from utils.log_utils import log_shard_progress

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smoltorrent")

API_BASE = "http://localhost:8000"


def gather_shards() -> dict:
    resp = httpx.post(f"{API_BASE}/gather-shards", timeout=300.0)
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
    logger.info("Triggering shard gather...")
    result = gather_shards()
    log_shard_progress(logger, result["gathered"], [])
    logger.info(f"All shards saved → {result.get('save_path')}")


if __name__ == "__main__":
    main()
