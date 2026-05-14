"""HTTP client helpers that call the local FastAPI server for store and gather operations."""
import logging

import httpx

from utils.log_utils import log_shard_progress

logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000"


def request_store_shards(model_id: str) -> dict:
    """POST /store-shard and return the parsed response body.

    Args:
        model_id: HuggingFace model ID passed as a query param to the API.

    Returns:
        Parsed JSON response dict with keys ``model_name``, ``num_shards``, ``sent_to``.

    Raises:
        httpx.HTTPStatusError: If the server returns a 4xx/5xx status.
    """
    resp = httpx.post(
        f"{API_BASE}/store-shard",
        params={"model_id": model_id},
        timeout=300.0,
    )
    try:
        body = resp.json()
    except Exception:
        resp.raise_for_status()
        raise
    if resp.is_error:
        for entry in body.get("detail", {}).get("errors", []):
            logger.error("  ✗ rank %s (%s): %s", entry["rank"], entry["host"], entry["error"])
        resp.raise_for_status()
    return body


def request_gather_shards(model_id: str) -> dict:
    """POST /gather-shards and return the parsed response body.

    Args:
        model_id: HuggingFace model ID passed as a query param to the API.

    Returns:
        Parsed JSON response dict with keys ``gathered`` and ``save_path``.

    Raises:
        httpx.HTTPStatusError: If the server returns a 4xx/5xx status.
    """
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
