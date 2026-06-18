"""Unit tests for the parallelised store-shard redundancy logic in backend/api.py.

Tests verify correct shard→worker assignment for REDUNDANCY=2 without needing
live Pi workers — all TCP calls are mocked.

Run with:  pytest test/test_store_redundancy_parallel.py -v
"""

import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1]))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_WORKERS = 4

_FAKE_WORKERS = [
    {"host": f"pi4-{r}", "ip": f"192.168.1.{r}", "rank": r, "port": 5000 + r}
    for r in range(1, N_WORKERS + 1)
]

_FAKE_CONFIG = {
    "ckpt_root": "/tmp/smoltorrent_test_ckpts",
    "devices_config": {"workers": _FAKE_WORKERS},
}


def _fake_tensors() -> dict:
    """4 small tensors so chunk_data splits cleanly into N_WORKERS shards."""
    return {f"layer.{i}.weight": torch.ones(8, 8) for i in range(N_WORKERS)}


@pytest.fixture()
def fake_ckpt(tmp_path):
    """Write a minimal safetensors file and return its path."""
    from safetensors.torch import save_file as st_save_file

    ckpt_dir = tmp_path / "run1" / "step_100"
    ckpt_dir.mkdir(parents=True)
    fpath = ckpt_dir / "model.safetensors"
    st_save_file(_fake_tensors(), str(fpath))
    return fpath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_body(response) -> str:
    return response.content.decode()


def _ok_send(worker, ckpt_path, file_offset, length, tensor_meta, rel_path, shard_filename, *args, **kwargs):
    return True, "", {"shard_path": f"/remote/worker_{worker['rank']}/{shard_filename}"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStoreRedundancyParallel:
    """Verify the parallelised two-round store sends to the right workers."""

    def _run_store(self, fake_ckpt, send_side_effect=None):
        """Mock all I/O, POST /store-shard, return (response_body, call_args_list)."""
        import backend.api as api_mod

        ckpt_root = str(fake_ckpt.parent.parent.parent)
        cfg = {**_FAKE_CONFIG, "ckpt_root": ckpt_root}
        send_mock = MagicMock(side_effect=send_side_effect or _ok_send)

        with (
            patch.object(api_mod, "load_config", return_value=cfg),
            patch.object(api_mod, "send_shard_to_worker", send_mock),
            patch.object(api_mod, "heartbeat_workers", return_value=[]),
            patch.object(api_mod, "add_shard_header"),
        ):
            client = TestClient(api_mod.app)
            resp = client.post("/store-shard", params={"ckpt_path": str(fake_ckpt)})

        return _collect_body(resp), send_mock.call_args_list

    # --- send count ---

    def test_total_sends_equals_workers_times_redundancy(self, fake_ckpt):
        body, calls = self._run_store(fake_ckpt)
        assert len(calls) == N_WORKERS * 2, (
            f"Expected {N_WORKERS * 2} sends, got {len(calls)}\n{body}"
        )

    # --- round 0: shard i → workers[i] (primary) ---

    def test_round0_shard_i_goes_to_worker_i(self, fake_ckpt):
        """Each shard's file_offset must be received by exactly 2 distinct workers."""
        import backend.api as api_mod

        ckpt_root = str(fake_ckpt.parent.parent.parent)
        cfg = {**_FAKE_CONFIG, "ckpt_root": ckpt_root}

        # (file_offset, worker_rank) pairs captured per call
        captured: list[tuple[int, int]] = []

        def _capture(worker, ckpt_path, file_offset, length, tensor_meta, rel_path, shard_filename, *args, **kwargs):
            captured.append((file_offset, worker["rank"]))
            return True, "", {"shard_path": "/tmp/x"}

        with (
            patch.object(api_mod, "load_config", return_value=cfg),
            patch.object(api_mod, "send_shard_to_worker", side_effect=_capture),
            patch.object(api_mod, "heartbeat_workers", return_value=[]),
            patch.object(api_mod, "add_shard_header"),
        ):
            client = TestClient(api_mod.app)
            client.post("/store-shard", params={"ckpt_path": str(fake_ckpt)})

        ranks_per_offset: dict[int, set[int]] = defaultdict(set)
        for offset, rank in captured:
            ranks_per_offset[offset].add(rank)

        # Every unique shard (identified by file_offset) must reach exactly 2 workers
        for offset, ranks in ranks_per_offset.items():
            assert len(ranks) == 2, (
                f"Shard at offset {offset} sent to {ranks} — expected 2 workers"
            )

    # --- round 1: shard i → workers[(i+1) % N] (replica) ---

    def test_replica_is_adjacent_worker(self, fake_ckpt):
        """The two workers that receive the same shard must be adjacent in the ring."""
        import backend.api as api_mod

        ckpt_root = str(fake_ckpt.parent.parent.parent)
        cfg = {**_FAKE_CONFIG, "ckpt_root": ckpt_root}

        ranks_per_offset: dict[int, list[int]] = defaultdict(list)

        def _capture(worker, ckpt_path, file_offset, length, tensor_meta, rel_path, shard_filename, *args, **kwargs):
            ranks_per_offset[file_offset].append(worker["rank"])
            return True, "", {"shard_path": "/tmp/x"}

        with (
            patch.object(api_mod, "load_config", return_value=cfg),
            patch.object(api_mod, "send_shard_to_worker", side_effect=_capture),
            patch.object(api_mod, "heartbeat_workers", return_value=[]),
            patch.object(api_mod, "add_shard_header"),
        ):
            client = TestClient(api_mod.app)
            client.post("/store-shard", params={"ckpt_path": str(fake_ckpt)})

        worker_ranks = [w["rank"] for w in _FAKE_WORKERS]
        for offset, ranks in ranks_per_offset.items():
            assert len(ranks) == 2, f"Expected 2 sends per shard at offset {offset}, got {ranks}"
            r0, r1 = ranks[0], ranks[1]
            idx0 = worker_ranks.index(r0)
            idx1 = worker_ranks.index(r1)
            adjacent = (idx1 == (idx0 + 1) % N_WORKERS) or (idx0 == (idx1 + 1) % N_WORKERS)
            assert adjacent, (
                f"Ranks {r0} and {r1} are not adjacent in the worker ring {worker_ranks}"
            )

    # --- response body content ---

    def test_body_contains_shard_filenames(self, fake_ckpt):
        """Both shard filenames (round 0 primary, round 1 replica) must appear in the body."""
        body, _ = self._run_store(fake_ckpt)
        assert "shard_0.safetensors" in body, f"Missing 'shard_0.safetensors' in:\n{body}"
        assert "shard_1.safetensors" in body, f"Missing 'shard_1.safetensors' in:\n{body}"

    def test_done_line_reports_correct_send_count(self, fake_ckpt):
        body, _ = self._run_store(fake_ckpt)
        expected = f"{N_WORKERS * 2}/{N_WORKERS * 2} sends"
        assert expected in body, f"Expected '{expected}' in body:\n{body}"

    def test_done_line_says_2x_replicated(self, fake_ckpt):
        body, _ = self._run_store(fake_ckpt)
        assert "2x replicated" in body, f"Missing '2x replicated' in:\n{body}"

    def test_no_permanently_failed_line(self, fake_ckpt):
        body, _ = self._run_store(fake_ckpt)
        assert "permanently failed" not in body

    def test_all_worker_ranks_appear_in_output(self, fake_ckpt):
        body, _ = self._run_store(fake_ckpt)
        for w in _FAKE_WORKERS:
            assert f"rank {w['rank']}" in body, (
                f"rank {w['rank']} not mentioned in body:\n{body}"
            )

    # --- failure + retry ---

    def test_failed_send_is_queued_for_retry(self, fake_ckpt):
        """If one send fails, the body must mention retry and ultimately not permanently fail."""
        call_count = {"n": 0}

        def _flaky(worker, ckpt_path, file_offset, length, tensor_meta, rel_path, shard_filename, *args, **kwargs):
            call_count["n"] += 1
            if worker["rank"] == _FAKE_WORKERS[0]["rank"] and call_count["n"] == 1:
                return False, "connection refused", {}
            return True, "", {"shard_path": "/tmp/x"}

        body, calls = self._run_store(fake_ckpt, send_side_effect=_flaky)
        assert "queuing retry" in body, f"Expected retry message:\n{body}"
        assert "permanently failed" not in body, f"Should have recovered:\n{body}"

    def test_no_error_in_body_on_all_success(self, fake_ckpt):
        body, _ = self._run_store(fake_ckpt)
        assert "ERROR" not in body, f"Unexpected ERROR:\n{body}"
