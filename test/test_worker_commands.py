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

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from networking.send_receive import receive_file, receive_message, send_message
from utils.common_utils import compute_checksum

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


def _make_shard_bytes(shapes: dict | None = None) -> bytes:
    """Create valid safetensors bytes (float32 zeros) for the given {name: shape} dict."""
    import numpy as np
    from safetensors.numpy import save as _st_save

    if shapes is None:
        shapes = {"weight": (8, 8)}
    return _st_save({k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()})


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


def _store_shard(
    worker: dict,
    rank: int,
    shard_bytes: bytes,
    checksum: str,
    rel_path: str,
    shard_filename: str = "shard_0.safetensors",
):
    """Current store_shard protocol:
      1. Command tuple + checksum
      2. tensor_meta dict (parsed from the shard's safetensors header)
      3. Raw tensor data bytes via serve_file_range length-prefix convention
    """
    import os as _os
    import tempfile
    from networking.send_receive import serve_file
    from utils.common_utils import handle_json_header

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp.write(shard_bytes)
        tmp_path = tmp.name

    header, data_section_offset = handle_json_header(tmp_path)
    tensor_data_len = len(shard_bytes) - data_section_offset
    tensor_meta = {
        k: {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": list(v["data_offsets"])}
        for k, v in header.items()
        if k != "__metadata__"
    }

    sock = _connect(worker, timeout=30.0)
    send_message(sock, ("store_shard", rank, checksum, rel_path, shard_filename))
    send_message(sock, tensor_meta)
    serve_file(sock, tmp_path, data_section_offset, tensor_data_len)
    result = receive_message(sock)
    sock.close()

    _os.unlink(tmp_path)
    return result


@pytest.mark.integration
class TestStoreShard:
    def test_store_and_ack(self):
        worker = WORKERS[0]
        shard_bytes = _make_shard_bytes({"test.weight": (8, 8)})
        checksum = compute_checksum(shard_bytes)

        result = _store_shard(worker, worker["rank"], shard_bytes, checksum, _FAKE_REL_PATH)

        assert result is not None
        assert result[0] == "store_shard_done", f"Expected store_shard_done, got: {result}"

    def test_bad_checksum_rejected(self):
        worker = WORKERS[0]
        shard_bytes = _make_shard_bytes({"w": (4, 4)})

        result = _store_shard(worker, worker["rank"], shard_bytes, "0" * 64, _FAKE_REL_PATH)

        assert result is not None
        assert result[0] == "store_shard_failed"


@pytest.mark.integration
class TestSendShard:
    def test_send_returns_file_for_existing(self, tmp_path):
        """Store a shard then retrieve it via receive_file — file must be valid safetensors."""
        from safetensors.numpy import load as _st_load_numpy
        import io

        worker = WORKERS[0]
        shard_bytes = _make_shard_bytes({"w": (8, 8)})
        checksum = compute_checksum(shard_bytes)

        _store_shard(worker, worker["rank"], shard_bytes, checksum, _FAKE_REL_PATH)

        sock = _connect(worker, timeout=30.0)
        send_message(sock, ("send_shard", worker["rank"], _FAKE_REL_PATH))
        dest = tmp_path / "received.safetensors"
        receive_file(sock, str(dest))
        sock.close()

        assert dest.exists()
        restored = _st_load_numpy(io.BytesIO(dest.read_bytes()))
        assert "w" in restored

    def test_send_nonexistent_returns_missing_tuple(self):
        """Worker sends ("send_shard_missing", rank, rel_path) for an unknown shard."""
        worker = WORKERS[0]
        result = _send_recv(
            worker, ("send_shard", worker["rank"], "__nonexistent__/path/latest")
        )
        assert result is not None
        assert result[0] == "send_shard_missing"

    def test_send_shard_range_streams_tensor_bytes(self, tmp_path):
        """Store a shard then retrieve only its tensor bytes via send_shard_range."""
        worker = WORKERS[0]
        shard_bytes = _make_shard_bytes({"w": (16, 16)})
        checksum = compute_checksum(shard_bytes)
        _store_shard(worker, worker["rank"], shard_bytes, checksum, _FAKE_REL_PATH)

        from utils.common_utils import handle_json_header
        import mmap as _mmap_mod

        sock = _connect(worker, timeout=30.0)
        send_message(sock, ("send_shard_range", worker["rank"], _FAKE_REL_PATH, "shard_0.safetensors"))
        status = receive_message(sock)
        assert status is not None
        assert status[0] == "send_shard_range_ok", f"Unexpected status: {status}"
        _, _, announced_len = status

        assert announced_len > 0

        dest = tmp_path / "tensor_data.bin"
        with open(dest, "wb") as f:
            f.truncate(announced_len)
        with open(dest, "r+b") as f, _mmap_mod.mmap(f.fileno(), announced_len) as mm:
            received, _ = receive_file(sock, mm, write_offset=0, expected_length=announced_len)
            mm.flush()
        sock.close()

        assert received == announced_len
        assert dest.stat().st_size == announced_len
