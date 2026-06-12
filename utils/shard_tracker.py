"""Lightweight shard-to-worker mapping — zero-dependency, JSON-backed.

Tracks which workers hold which shards so the master can gather only from
relevant peers instead of broadcasting to everyone.  Survives restarts via
atomic file writes (``os.replace``).

Usage (master side only — workers don't need this)::

    from utils.shard_tracker import add_shard, get_ranks, remove_worker

    add_shard(rank=3, shard_key="Qwen2.5-0.5B/step_1000")
    ranks = get_ranks("Qwen2.5-0.5B/step_1000")   # → [3]
    remove_worker(rank=3)
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_TRACKER_PATH = Path(__file__).resolve().parents[1] / "shard_map.json"
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load() -> dict[str, list[int]]:
    """Read the current map from disk (or return empty dict if no file)."""
    if not _TRACKER_PATH.exists():
        return {}
    try:
        with open(_TRACKER_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict[str, list[int]]) -> None:
    """Atomically write *data* to disk (tmp → rename)."""
    tmp = _TRACKER_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, _TRACKER_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_shard(rank: int, shard_key: str) -> None:
    """Record that *rank* holds *shard_key*.

    Idempotent — calling twice with the same (rank, shard_key) does
    not create duplicate entries.

    Args:
        rank:      Worker rank (integer, matches config.yaml).
        shard_key: Unique identifier for this shard, e.g.
                   ``"mlx-community--Qwen2.5-0.5B/step_1000"``.
    """
    with _lock:
        data = _load()
        ranks = data.setdefault(shard_key, [])
        if rank not in ranks:
            ranks.append(rank)
        _save(data)


def get_ranks(shard_key: str) -> list[int]:
    """Return the list of worker ranks that hold *shard_key*.

    Args:
        shard_key: Shard identifier (see :func:`add_shard`).

    Returns:
        List of integer ranks (may be empty if nobody has registered
        this shard yet).
    """
    with _lock:
        return list(_load().get(shard_key, []))


def remove_worker(rank: int) -> None:
    """Remove *rank* from every shard entry.

    Call when a worker disconnects or is shut down gracefully.

    Args:
        rank: Worker rank to purge from the map.
    """
    with _lock:
        data = _load()
        changed = False
        for ranks in data.values():
            while rank in ranks:
                ranks.remove(rank)
                changed = True
        if changed:
            _save(data)


def list_shards_for_rank(rank: int) -> list[str]:
    """Return every shard key that *rank* holds.

    Args:
        rank: Worker rank.

    Returns:
        List of shard key strings.
    """
    with _lock:
        data = _load()
        return [k for k, ranks in data.items() if rank in ranks]


def clear() -> None:
    """Drop the entire shard map (useful for testing / reset)."""
    with _lock:
        if _TRACKER_PATH.exists():
            _TRACKER_PATH.unlink()
