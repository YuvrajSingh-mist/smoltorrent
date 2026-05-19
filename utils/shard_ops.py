"""HTTP client helpers that call the local FastAPI server for store and gather operations."""

import logging

import httpx

logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000"


def request_store_shards(ckpt_path: str, log_fn=logger.info) -> None:
    """Stream log lines from POST /store-shard and forward each to ``log_fn``.

    Args:
        ckpt_path: Absolute path to the checkpoint .safetensors file on master.
        log_fn: Callable that accepts a single string — defaults to this module's logger.

    Raises:
        RuntimeError: If the server reports an error (line starting with ``ERROR:``).
        httpx.HTTPStatusError: If the HTTP connection itself fails.
    """
    with httpx.stream(
        "POST",
        f"{API_BASE}/store-shard",
        params={"ckpt_path": ckpt_path},
        timeout=None,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            log_fn(line)
            if line.startswith("ERROR:"):
                raise RuntimeError(line)


def request_gather_shards(ckpt_path: str, log_fn=logger.info) -> None:
    """Stream log lines from POST /gather-shards and forward each to ``log_fn``.

    Args:
        ckpt_path: Absolute path to the checkpoint file (same path used for store).
        log_fn: Callable that accepts a single string — defaults to this module's logger.

    Raises:
        RuntimeError: If the server reports an error (line starting with ``ERROR:``).
        httpx.HTTPStatusError: If the HTTP connection itself fails.
    """
    with httpx.stream(
        "POST",
        f"{API_BASE}/gather-shards",
        params={"ckpt_path": ckpt_path},
        timeout=None,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            log_fn(line)
            if line.startswith("ERROR:"):
                raise RuntimeError(line)
