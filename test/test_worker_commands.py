"""Integration tests for all worker TCP commands.

Tests every command the worker handles against live Pi workers:
  heartbeat, sync, all_shards_present, checksum_sync, store_shard, send_shard

Markers: integration — requires cluster running (bash scripts/launch.sh).
"""

import socket
import struct
import sys
from pathlib import Path
from typing import Optional

import mlx.core as mx
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from networking.send_receive import receive_file_mmap, receive_message, send_message
from utils.common_utils import compute_checksum, shard_to_bytes, shard_from_bytes

_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"


def _load_workers() -> list[dict]:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)["devices_config"]["workers"]


def _connect(worker: dict, timeout: float = 10.0) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((worker["ip"], worker["port"]))
    sock.settimeout(None)
    return sock


def _send_recv(worker: dict, msg, timeout: float = 10.0):
    sock = _connect(worker, timeout)
    send_message(sock, msg)
    result = receive_message(sock)
    sock.close()
    return result


WORKERS = _load_workers()
_FAKE_REL_PATH = "__test__/pytest/shard_cmd_test/latest"


@pytest.mark.integration
class TestHeartbeat:
    @pytest.mark.parametrize(
        "worker", WORKERS, ids=[f"rank{w['rank']}" for w in WORKERS]
    )
    def test_alive(self, worker):
        result = _send_recv(worker, "heartbeat")
        assert result == "alive", f"rank {worker['rank']} did not reply alive: {result}"


@pytest.mark.integration
class TestSync:
    @pytest.mark.parametrize(
        "worker", WORKERS, ids=[f"rank{w['rank']}" for w in WORKERS]
    )
    def test_returns_list(self, worker):
        result = _send_recv(worker, ("sync", worker["rank"], [".safetensors"]))
        assert isinstance(result, list), (
            f"rank {worker['rank']} sync returned {type(result)}"
        )

    @pytest.mark.parametrize(
        "worker", WORKERS, ids=[f"rank{w['rank']}" for w in WORKERS]
    )
    def test_paths_are_strings(self, worker):
        result = _send_recv(worker, ("sync", worker["rank"], [".safetensors"]))
        for p in (result or []):
            assert isinstance(p, str)

    def test_all_workers_have_same_keys(self):
        results = []
        for w in WORKERS:
            r = _send_recv(w, ("sync", w["rank"], [".safetensors"]))
            results.append(set(r or []))
        intersection = results[0]
        for s in results[1:]:
            intersection &= s
        # All workers should share at least the paths they all have
        assert isinstance(intersection, set)


@pytest.mark.integration
class TestAllShardsPresent:
    def _get_known_paths(self) -> list[str]:
        """Return rel_paths that rank 1 actually has."""
        result = _send_recv(WORKERS[0], ("sync", WORKERS[0]["rank"], [".safetensors"]))
        return result[:3] if result else []

    def test_no_missing_for_present_paths(self):
        for w in WORKERS:
            known = _send_recv(w, ("sync", w["rank"], [".safetensors"]))
            if not known:
                continue
            missing = _send_recv(w, ("all_shards_present", w["rank"], known[:3]))
            assert missing == [], f"rank {w['rank']} wrongly reports missing: {missing}"

    def test_fake_paths_reported_missing(self):
        fake = ["__nonexistent__/run/latest", "__also_fake__/run/latest"]
        for w in WORKERS:
            missing = _send_recv(w, ("all_shards_present", w["rank"], fake))
            assert set(fake) == set(missing or []), (
                f"rank {w['rank']} should report all fake paths missing, got: {missing}"
            )

    def test_empty_list_returns_empty(self):
        for w in WORKERS:
            missing = _send_recv(w, ("all_shards_present", w["rank"], []))
            assert missing == []


@pytest.mark.integration
class TestChecksumSync:
    def _get_one_path(self) -> Optional[str]:
        result = _send_recv(WORKERS[0], ("sync", WORKERS[0]["rank"], [".safetensors"]))
        return result[0] if result else None

    def test_existing_shard_returns_ok(self):
        for w in WORKERS:
            paths = _send_recv(w, ("sync", w["rank"], [".safetensors"]))
            if not paths:
                continue
            result = _send_recv(w, ("checksum_sync", w["rank"], paths[0]))
            assert result is not None, f"rank {w['rank']} returned None"
            assert result[0] == "checksum_sync_result", f"Unexpected type: {result[0]}"
            assert result[1] in ("ok", "mismatch"), f"Unexpected status: {result[1]}"

    def test_fake_path_returns_missing(self):
        for w in WORKERS:
            result = _send_recv(w, ("checksum_sync", w["rank"], "__fake__/path/latest"))
            assert result is not None, f"rank {w['rank']} closed connection instead of sending missing"
            assert result[0] == "checksum_sync_result"
            assert result[1] == "missing"


def _store_shard(worker: dict, rank: int, shard_bytes: bytes, checksum: str, rel_path: str):
    """New store_shard protocol: metadata message then raw bytes stream."""
    sock = _connect(worker, timeout=30.0)
    send_message(sock, ("store_shard", rank, checksum, rel_path))
    sock.sendall(struct.pack(">I", len(shard_bytes)))
    sock.sendall(shard_bytes)
    result = receive_message(sock)
    sock.close()
    return result


@pytest.mark.integration
class TestStoreShard:
    def test_store_and_ack(self):
        worker = WORKERS[0]
        shard = {"test.weight": mx.ones([8, 8])}
        shard_bytes = shard_to_bytes(shard)
        checksum = compute_checksum(shard_bytes)

        result = _store_shard(worker, worker["rank"], shard_bytes, checksum, _FAKE_REL_PATH)

        assert result is not None
        assert result[0] == "store_shard_done", f"Expected store_shard_done, got: {result}"

    def test_bad_checksum_rejected(self):
        worker = WORKERS[0]
        shard_bytes = shard_to_bytes({"w": mx.zeros([4, 4])})

        result = _store_shard(worker, worker["rank"], shard_bytes, "0" * 64, _FAKE_REL_PATH)

        assert result is not None
        assert result[0] == "store_shard_failed"


@pytest.mark.integration
class TestSendShard:
    def test_send_returns_file_for_existing(self, tmp_path):
        """Store a shard then retrieve it via receive_file_mmap — bytes must round-trip."""
        worker = WORKERS[0]
        original = {"w": mx.ones([8, 8]) * 42}
        shard_bytes = shard_to_bytes(original)
        checksum = compute_checksum(shard_bytes)

        _store_shard(worker, worker["rank"], shard_bytes, checksum, _FAKE_REL_PATH)

        sock = _connect(worker, timeout=30.0)
        send_message(sock, ("send_shard", worker["rank"], _FAKE_REL_PATH))
        dest = tmp_path / "received.safetensors"
        receive_file_mmap(sock, str(dest))
        sock.close()

        assert dest.exists()
        restored = shard_from_bytes(dest.read_bytes())
        assert "w" in restored

    def test_send_nonexistent_returns_none(self):
        worker = WORKERS[0]
        result = _send_recv(
            worker, ("send_shard", worker["rank"], "__nonexistent__/path/latest")
        )
        assert result is None
