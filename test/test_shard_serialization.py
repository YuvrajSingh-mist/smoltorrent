"""Unit tests for shard serialization, checksum, and merge.

No network, no cluster. Tests the core data pipeline:
  chunk_data → shard_to_bytes → shard_from_bytes → merge_shards

Markers: (none) — always runs.
"""
import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.common_utils import chunk_data, compute_checksum, merge_shards, shard_from_bytes, shard_to_bytes


def _make_tensors(n: int = 20) -> dict:
    """Small synthetic tensor dict (MLX arrays, float32)."""
    return {f"layer_{i}.weight": mx.ones([4, 4]) * i for i in range(n)}


class TestShardToBytes:
    def test_returns_bytes(self):
        shard = _make_tensors(4)
        assert isinstance(shard_to_bytes(shard), bytes)

    def test_nonempty(self):
        assert len(shard_to_bytes(_make_tensors(4))) > 0

    def test_deterministic(self):
        shard = _make_tensors(4)
        assert shard_to_bytes(shard) == shard_to_bytes(shard)

    def test_different_shards_differ(self):
        a = {"w": mx.ones([4, 4])}
        b = {"w": mx.zeros([4, 4])}
        assert shard_to_bytes(a) != shard_to_bytes(b)


class TestShardFromBytes:
    def test_returns_dict(self):
        b = shard_to_bytes(_make_tensors(4))
        result = shard_from_bytes(b)
        assert isinstance(result, dict)

    def test_keys_preserved(self):
        tensors = _make_tensors(4)
        result = shard_from_bytes(shard_to_bytes(tensors))
        assert set(result.keys()) == set(tensors.keys())

    def test_values_correct(self):
        tensors = _make_tensors(4)
        result = shard_from_bytes(shard_to_bytes(tensors))
        for k, v in tensors.items():
            mx.eval(v)
            restored = result[k]
            if hasattr(restored, "numpy"):
                import numpy as np
                assert np.allclose(v.tolist(), restored.numpy()), f"Mismatch on {k}"
            else:
                assert v.tolist() == restored.tolist(), f"Mismatch on {k}"


class TestRoundTrip:
    def test_single_shard_roundtrip(self):
        tensors = _make_tensors(8)
        restored = shard_from_bytes(shard_to_bytes(tensors))
        assert set(restored.keys()) == set(tensors.keys())

    def test_chunk_and_reassemble(self):
        tensors = _make_tensors(20)
        chunks = chunk_data(tensors, n_chunks=4)
        assert len(chunks) == 4

        # Serialize + deserialize each chunk
        restored_chunks = [shard_from_bytes(shard_to_bytes(chunks[i])) for i in range(4)]
        merged = merge_shards(restored_chunks)

        assert set(merged.keys()) == set(tensors.keys())

    def test_no_key_loss_across_chunks(self):
        tensors = _make_tensors(16)
        chunks = chunk_data(tensors, n_chunks=4)
        all_keys = set()
        for chunk in chunks.values():
            all_keys.update(chunk.keys())
        assert all_keys == set(tensors.keys())


class TestChecksum:
    def test_returns_hex_string(self):
        cksum = compute_checksum(b"hello")
        assert isinstance(cksum, str)
        assert len(cksum) == 64  # SHA-256 hex

    def test_deterministic(self):
        data = b"test data"
        assert compute_checksum(data) == compute_checksum(data)

    def test_different_bytes_differ(self):
        assert compute_checksum(b"abc") != compute_checksum(b"xyz")

    def test_shard_checksum_stable(self):
        shard = _make_tensors(4)
        b = shard_to_bytes(shard)
        assert compute_checksum(b) == compute_checksum(b)

    def test_modified_shard_different_checksum(self):
        a = shard_to_bytes({"w": mx.ones([4, 4])})
        b = shard_to_bytes({"w": mx.zeros([4, 4])})
        assert compute_checksum(a) != compute_checksum(b)


class TestMergeShards:
    def test_merges_two_dicts(self):
        a = {"layer_0": mx.ones([2, 2])}
        b = {"layer_1": mx.zeros([2, 2])}
        merged = merge_shards([a, b])
        assert "layer_0" in merged
        assert "layer_1" in merged

    def test_all_keys_present(self):
        tensors = _make_tensors(12)
        chunks = chunk_data(tensors, n_chunks=4)
        merged = merge_shards(list(chunks.values()))
        assert set(merged.keys()) == set(tensors.keys())

    def test_single_shard_unchanged(self):
        tensors = {"w": mx.ones([3, 3])}
        merged = merge_shards([tensors])
        assert set(merged.keys()) == {"w"}

    def test_empty_list(self):
        assert merge_shards([]) == {}
