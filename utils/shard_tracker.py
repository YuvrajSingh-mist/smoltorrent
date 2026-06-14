"""Lightweight shard-to-worker mapping — SQLite-backed.

Tracks which workers hold which shards and stores the full safetensors header
so gather can reconstruct the merged file without querying workers or needing
the original checkpoint on disk.

DB tables::

    shards        — one row per (shard_key, rank, shard_file); shard_index records
                    which chunk of the model data (0, 1, ...) each row covers.
    shard_headers — one row per shard_key, full safetensors layout

The shard_index column is the key to correct gather fallback:
  - shard_index 0 = first chunk of tensor data
  - shard_index 1 = second chunk, etc.

For REDUNDANCY=2 with 2 workers, the layout is:
  (shard_key, shard_index=0, rank=0, shard_file=shard_0.safetensors)  ← primary
  (shard_key, shard_index=0, rank=1, shard_file=shard_1.safetensors)  ← replica
  (shard_key, shard_index=1, rank=1, shard_file=shard_0.safetensors)  ← primary
  (shard_key, shard_index=1, rank=0, shard_file=shard_1.safetensors)  ← replica

get_replica_map(shard_key) returns the full redundancy map in a single DB query,
ordered primary-first per shard_index, so gather can pre-load all fallback routes.

Usage (master side only — workers don't need this)::

    from utils.shard_tracker import (
        add_shard, add_shard_header,
        get_ranks, get_shard_header, get_replica_map,
    )

    add_shard(rank=0, shard_index=0, shard_key="Qwen2.5-0.5B/step_1000",
              host="minilab-pi4-4", shard_files=["shard_0.safetensors"],
              size_bytes=989855744, source_path="/path/to/model.safetensors")
    add_shard_header(shard_key="Qwen2.5-0.5B/step_1000",
                     header_json='{"model.weight": {...}}',
                     data_section_offset=1024, num_workers=2)
    ranks      = get_ranks("Qwen2.5-0.5B/step_1000")           # → [0, 1]
    replica_map = get_replica_map("Qwen2.5-0.5B/step_1000")
                  # → {0: [{"rank": 0, ...}, {"rank": 1, ...}], 1: [...]}
    info       = get_shard_header("Qwen2.5-0.5B/step_1000")
                  # → {header_json, data_section_offset, num_workers}
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH   = Path(__file__).resolve().parents[1] / "shard_map.db"
_JSON_PATH = Path(__file__).resolve().parents[1] / "shard_map.json"  # legacy — migrated on first use


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS shards (
            shard_key    TEXT    NOT NULL,
            rank         INTEGER NOT NULL,
            host         TEXT    DEFAULT '',
            shard_file   TEXT    DEFAULT '',
            stored_at    TEXT    DEFAULT '',
            size_bytes   INTEGER DEFAULT 0,
            source_path  TEXT    DEFAULT '',
            shard_index  INTEGER DEFAULT -1,
            checksum     TEXT    DEFAULT '',
            PRIMARY KEY (shard_key, rank, shard_file)
        );
        CREATE TABLE IF NOT EXISTS shard_headers (
            shard_key            TEXT    PRIMARY KEY,
            header_json          TEXT    NOT NULL,
            data_section_offset  INTEGER NOT NULL,
            num_workers          INTEGER NOT NULL,
            stored_at            TEXT    DEFAULT '',
            shard_ranges_json    TEXT    DEFAULT '',
            total_tensor_bytes   INTEGER DEFAULT 0,
            original_checksum    TEXT    DEFAULT ''
        );
    """)
    # Column + index migrations for DBs created before these fields existed.
    # Index must come after the column ALTER so it runs on an up-to-date schema.
    migrations = [
        "ALTER TABLE shards ADD COLUMN shard_index INTEGER DEFAULT -1",
        "ALTER TABLE shards ADD COLUMN checksum TEXT DEFAULT ''",
        "ALTER TABLE shard_headers ADD COLUMN shard_ranges_json TEXT DEFAULT ''",
        "ALTER TABLE shard_headers ADD COLUMN total_tensor_bytes INTEGER DEFAULT 0",
        "ALTER TABLE shard_headers ADD COLUMN original_checksum TEXT DEFAULT ''",
        "CREATE INDEX IF NOT EXISTS idx_shards_shard_index ON shards (shard_key, shard_index)",
    ]
    for sql in migrations:
        try:
            conn.execute(sql)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column/index already exists



def _db() -> sqlite3.Connection:
    """Open connection, ensure schema, run one-time JSON migration if needed."""
    conn = _connect()
    _ensure_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# Public API — shard placement
# ---------------------------------------------------------------------------

def add_shard(
    rank: int,
    shard_key: str,
    host: str = "",
    shard_files: list[str] | None = None,
    size_bytes: int = 0,
    source_path: str = "",
    shard_index: int = -1,
    checksum: str = "",
) -> None:
    """Record that *rank* holds *shard_key*, updating metadata in place.

    Idempotent — safe to call multiple times for the same (rank, shard_key).

    Args:
        rank:        Worker rank (integer, matches config.yaml).
        shard_key:   Unique identifier, e.g. ``"Qwen2.5-0.5B/step_1000"``.
        host:        Human-readable hostname of the worker.
        shard_files: Filenames stored on this worker (e.g. ``["shard_0.safetensors"]``).
        size_bytes:  Total checkpoint size in bytes (informational).
        source_path: Absolute path to the original checkpoint file on the coordinator.
        shard_index: Which chunk of model data this covers (0, 1, ..., n-1).
                     -1 means unknown (legacy rows from before this field existed).
        checksum:    SHA-256 of this shard's raw tensor bytes (coordinator-computed,
                     same bytes the worker received). Used for per-shard integrity
                     checks at gather time.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files = shard_files or [""]
    conn = _db()
    with conn:
        for sf in files:
            conn.execute(
                """INSERT INTO shards
                       (shard_key, rank, host, shard_file, stored_at, size_bytes, source_path, shard_index, checksum)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(shard_key, rank, shard_file) DO UPDATE SET
                       host=excluded.host,
                       stored_at=excluded.stored_at,
                       size_bytes=CASE WHEN excluded.size_bytes > 0 THEN excluded.size_bytes ELSE size_bytes END,
                       source_path=CASE WHEN excluded.source_path != '' THEN excluded.source_path ELSE source_path END,
                       shard_index=CASE WHEN excluded.shard_index >= 0 THEN excluded.shard_index ELSE shard_index END,
                       checksum=CASE WHEN excluded.checksum != '' THEN excluded.checksum ELSE checksum END""",
                (shard_key, rank, host, sf, now, size_bytes, source_path, shard_index, checksum),
            )
    conn.close()
    logger.info(
        "[tracker] add_shard shard_key=%s rank=%d shard_index=%d host=%s files=%s checksum=%s…",
        shard_key, rank, shard_index, host, files, checksum[:16] if checksum else "",
    )


def get_ranks(shard_key: str) -> list[int]:
    """Return the list of worker ranks that hold *shard_key*."""
    conn = _db()
    rows = conn.execute(
        "SELECT DISTINCT rank FROM shards WHERE shard_key=? ORDER BY rank", (shard_key,)
    ).fetchall()
    conn.close()
    ranks = [r["rank"] for r in rows]
    logger.debug("[tracker] get_ranks shard_key=%s → %s", shard_key, ranks)
    return ranks



def get_replica_map(shard_key: str) -> dict[int, list[dict]]:
    """Fetch the full redundancy map for *shard_key* in a single DB query.

    Returns a dict mapping each shard_index to its ordered replica list, so
    gather can pre-load all fallback routes before spawning threads — no
    per-thread DB calls needed.

    Returns::

        {
            0: [{"rank": 0, "shard_file": "shard_0.safetensors"},   # primary
                {"rank": 1, "shard_file": "shard_1.safetensors"}],  # replica
            1: [{"rank": 1, "shard_file": "shard_0.safetensors"},
                {"rank": 0, "shard_file": "shard_1.safetensors"}],
        }

    Only includes rows with shard_index >= 0 (excludes legacy rows with shard_index = -1).
    Returns an empty dict if nothing is tracked or all rows are legacy.
    """
    conn = _db()
    rows = conn.execute(
        """SELECT shard_index, rank, shard_file, checksum FROM shards
           WHERE shard_key=? AND shard_index >= 0
           ORDER BY shard_index ASC, shard_file ASC""",
        (shard_key,),
    ).fetchall()
    conn.close()

    replica_map: dict[int, list[dict]] = {}
    for r in rows:
        idx = r["shard_index"]
        replica_map.setdefault(idx, []).append({
            "rank": r["rank"],
            "shard_file": r["shard_file"],
            "checksum": r["checksum"],
        })

    logger.info(
        "[tracker] get_replica_map shard_key=%s → %d shard_index(es) tracked",
        shard_key, len(replica_map),
    )
    return replica_map


def get_shard_info(shard_key: str) -> dict | None:
    """Return the full metadata dict for *shard_key*, or None if not tracked.

    Returns the same structure as the old JSON map for backward compatibility.
    """
    conn = _db()
    rows = conn.execute(
        "SELECT rank, host, shard_file, stored_at, size_bytes, source_path FROM shards WHERE shard_key=?",
        (shard_key,),
    ).fetchall()
    conn.close()
    if not rows:
        return None

    ranks, hosts, shard_files = [], {}, []
    stored_at, size_bytes, source_path = "", 0, ""
    for r in rows:
        if r["rank"] not in ranks:
            ranks.append(r["rank"])
        if r["host"]:
            hosts[str(r["rank"])] = r["host"]
        if r["shard_file"] and r["shard_file"] not in shard_files:
            shard_files.append(r["shard_file"])
        stored_at  = r["stored_at"]
        size_bytes = r["size_bytes"]
        source_path = r["source_path"]

    return {
        "ranks": sorted(ranks),
        "hosts": hosts,
        "stored_at": stored_at,
        "size_bytes": size_bytes,
        "shard_files": shard_files,
        "source_path": source_path,
    }



def list_shards_for_rank(rank: int) -> list[str]:
    """Return every shard key that *rank* holds."""
    conn = _db()
    rows = conn.execute(
        "SELECT DISTINCT shard_key FROM shards WHERE rank=?", (rank,)
    ).fetchall()
    conn.close()
    return [r["shard_key"] for r in rows]



# ---------------------------------------------------------------------------
# Public API — safetensors header storage
# ---------------------------------------------------------------------------

def add_shard_header(
    shard_key: str,
    header_json: str,
    data_section_offset: int,
    num_workers: int,
    shard_ranges: list[dict] | None = None,
    total_tensor_bytes: int = 0,
    original_checksum: str = "",
) -> None:
    """Store the full safetensors header and precomputed layout for *shard_key*.

    Called at store time so gather can reconstruct the merged file without
    re-parsing the header or re-running get_shard_ranges.

    Args:
        shard_key:           Matches the key used in add_shard.
        header_json:         Raw JSON string from the original checkpoint header.
        data_section_offset: Byte offset where tensor data starts in the original file.
        num_workers:         Number of shards the checkpoint was split into.
        shard_ranges:        List of {"file_offset": int, "length": int} per shard.
                             Stored as JSON so gather skips get_shard_ranges() entirely.
        total_tensor_bytes:  Sum of all shard lengths — avoids re-summing at gather.
        original_checksum:   SHA-256 of the original file's full tensor data section.
                             Used at gather time to verify the merged file is byte-perfect.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ranges_json = json.dumps(shard_ranges) if shard_ranges else ""
    conn = _db()
    with conn:
        conn.execute(
            """INSERT INTO shard_headers
                   (shard_key, header_json, data_section_offset, num_workers, stored_at,
                    shard_ranges_json, total_tensor_bytes, original_checksum)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(shard_key) DO UPDATE SET
                   header_json=excluded.header_json,
                   data_section_offset=excluded.data_section_offset,
                   num_workers=excluded.num_workers,
                   stored_at=excluded.stored_at,
                   shard_ranges_json=CASE WHEN excluded.shard_ranges_json != '' THEN excluded.shard_ranges_json ELSE shard_ranges_json END,
                   total_tensor_bytes=CASE WHEN excluded.total_tensor_bytes > 0 THEN excluded.total_tensor_bytes ELSE total_tensor_bytes END,
                   original_checksum=CASE WHEN excluded.original_checksum != '' THEN excluded.original_checksum ELSE original_checksum END""",
            (shard_key, header_json, data_section_offset, num_workers, now,
             ranges_json, total_tensor_bytes, original_checksum),
        )
    conn.close()
    logger.info(
        "[tracker] add_shard_header shard_key=%s data_section_offset=%d num_workers=%d "
        "total_tensor_bytes=%d original_checksum=%s…",
        shard_key, data_section_offset, num_workers,
        total_tensor_bytes, original_checksum[:16] if original_checksum else "",
    )


def get_shard_header(shard_key: str) -> dict | None:
    """Return the stored safetensors header for *shard_key*, or None if not stored.

    Returns::

        {
            "header_json":          str,   # raw JSON from original checkpoint
            "data_section_offset":  int,   # byte offset where tensors start
            "num_workers":          int,
            "stored_at":            str,
        }
    """
    conn = _db()
    row = conn.execute(
        """SELECT header_json, data_section_offset, num_workers, stored_at,
                  shard_ranges_json, total_tensor_bytes, original_checksum
           FROM shard_headers WHERE shard_key=?""",
        (shard_key,),
    ).fetchone()
    conn.close()
    if row is None:
        logger.debug("[tracker] get_shard_header shard_key=%s → not found", shard_key)
        return None
    logger.debug(
        "[tracker] get_shard_header shard_key=%s data_section_offset=%d num_workers=%d total_tensor_bytes=%d",
        shard_key, row["data_section_offset"], row["num_workers"], row["total_tensor_bytes"],
    )
    shard_ranges = json.loads(row["shard_ranges_json"]) if row["shard_ranges_json"] else None
    return {
        "header_json":         row["header_json"],
        "data_section_offset": row["data_section_offset"],
        "num_workers":         row["num_workers"],
        "stored_at":           row["stored_at"],
        "shard_ranges":        shard_ranges,
        "total_tensor_bytes":  row["total_tensor_bytes"],
        "original_checksum":   row["original_checksum"],
    }
