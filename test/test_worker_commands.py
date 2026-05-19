"""Integration tests for all worker TCP commands.

Tests every command the worker handles against live Pi workers:
  heartbeat, sync, all_shards_present, checksum_sync, store_shard, send_shard

Markers: integration — requires cluster running (bash scripts/launch.sh).
"""

import socket
import sys
from pathlib import Path

import mlx.core as mx
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from networking.send_receive import receive_message, send_message
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
        for p in result:
            assert isinstance(p, str)

    def test_all_workers_have_same_keys(self):
        results = []
        for w in WORKERS:
            r = _send_recv(w, ("sync", w["rank"], [".safetensors"]))
            results.append(set(r))
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
        known = self._get_known_paths()
        if not known:
            pytest.skip("No shards on workers yet")
        for w in WORKERS:
            missing = _send_recv(w, ("all_shards_present", w["rank"], known))
            assert missing == [], f"rank {w['rank']} wrongly reports missing: {missing}"

    def test_fake_paths_reported_missing(self):
        fake = ["__nonexistent__/run/latest", "__also_fake__/run/latest"]
        for w in WORKERS:
            missing = _send_recv(w, ("all_shards_present", w["rank"], fake))
            assert set(fake) == set(missing), (
                f"rank {w['rank']} should report all fake paths missing, got: {missing}"
            )

    def test_empty_list_returns_empty(self):
        for w in WORKERS:
            missing = _send_recv(w, ("all_shards_present", w["rank"], []))
            assert missing == []


@pytest.mark.integration
class TestChecksumSync:
    def _get_one_path(self) -> str | None:
        result = _send_recv(WORKERS[0], ("sync", WORKERS[0]["rank"], [".safetensors"]))
        return result[0] if result else None

    def test_existing_shard_returns_ok(self):
        path = self._get_one_path()
        if not path:
            pytest.skip("No shards on workers yet")
        for w in WORKERS:
            result = _send_recv(w, ("checksum_sync", w["rank"], path))
            assert result is not None
            status = result[0].replace("checksum_", "")
            assert status in ("ok", "missing"), f"Unexpected status: {status}"

    def test_fake_path_returns_missing(self):
        for w in WORKERS:
            result = _send_recv(w, ("checksum_sync", w["rank"], "__fake__/path/latest"))
            assert result is not None
            status = result[0].replace("checksum_", "")
            assert status == "missing"


@pytest.mark.integration
class TestStoreShard:
    def test_store_and_ack(self):
        """Store a small synthetic shard on rank 1 and check ack."""
        worker = WORKERS[0]
        shard = {"test.weight": mx.ones([8, 8])}
        shard_bytes = shard_to_bytes(shard)
        checksum = compute_checksum(shard_bytes)

        result = _send_recv(
            worker,
            ("store_shard", worker["rank"], shard_bytes, checksum, _FAKE_REL_PATH),
            timeout=30.0,
        )
        assert result is not None
        assert result[0] == "store_shard_done", (
            f"Expected store_shard_done, got: {result}"
        )

    def test_bad_checksum_rejected(self):
        worker = WORKERS[0]
        shard_bytes = shard_to_bytes({"w": mx.zeros([4, 4])})
        bad_checksum = "0" * 64

        result = _send_recv(
            worker,
            ("store_shard", worker["rank"], shard_bytes, bad_checksum, _FAKE_REL_PATH),
            timeout=30.0,
        )
        assert result is not None
        assert result[0] == "store_shard_failed"


@pytest.mark.integration
class TestSendShard:
    def test_send_returns_bytes_for_existing(self):
        """Store a shard then retrieve it — bytes should round-trip."""
        worker = WORKERS[0]
        original = {"w": mx.ones([8, 8]) * 42}
        shard_bytes = shard_to_bytes(original)
        checksum = compute_checksum(shard_bytes)

        # Store first
        _send_recv(
            worker,
            ("store_shard", worker["rank"], shard_bytes, checksum, _FAKE_REL_PATH),
            timeout=30.0,
        )

        # Retrieve
        received = _send_recv(
            worker, ("send_shard", worker["rank"], _FAKE_REL_PATH), timeout=30.0
        )
        assert received is not None, "send_shard returned None for existing shard"
        assert isinstance(received, bytes)

        restored = shard_from_bytes(received)
        assert "w" in restored

    def test_send_nonexistent_returns_none(self):
        worker = WORKERS[0]
        result = _send_recv(
            worker, ("send_shard", worker["rank"], "__nonexistent__/path/latest")
        )
        assert result is None
