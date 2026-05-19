"""Unit tests for the parallelised store-shard redundancy logic in backend/api.py.

Tests verify correct shard→worker assignment for REDUNDANCY=2 without needing
live Pi workers — all TCP calls are mocked.

Run with:  pytest test/test_store_redundancy_parallel.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

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
    """Read a streaming TestClient response into one string."""
    return response.content.decode()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStoreRedundancyParallel:
    """Verify that the parallelised two-round store sends to the right workers."""

    def _run_store(self, fake_ckpt, send_side_effect=None):
        """
        Import the FastAPI app, mock out all I/O, POST /store-shard, and return
        (response_body, list_of_send_calls).

        send_side_effect: optional callable or list to override _send_shard_to_worker.
        """
        import backend.api as api_mod

        ckpt_root = str(fake_ckpt.parent.parent.parent)  # /tmp/.../
        cfg = {**_FAKE_CONFIG, "ckpt_root": ckpt_root}

        # Default: every send succeeds immediately
        def _ok_send(worker, shard_bytes, checksum, rel_path):
            return True, "", {"shard_path": f"/remote/shard_{worker['rank']}.safetensors"}

        side_effect = send_side_effect or _ok_send

        send_mock = MagicMock(side_effect=side_effect)

        with patch.object(api_mod, "_load_config", return_value=cfg), \
             patch.object(api_mod, "_send_shard_to_worker", send_mock):
            client = TestClient(api_mod.app)
            resp = client.post("/store-shard", params={"ckpt_path": str(fake_ckpt)})

        body = _collect_body(resp)
        return body, send_mock.call_args_list

    # --- correctness of the round assignments ---

    def test_total_sends_equals_workers_times_redundancy(self, fake_ckpt):
        body, calls = self._run_store(fake_ckpt)
        assert len(calls) == N_WORKERS * 2, (
            f"Expected {N_WORKERS * 2} sends, got {len(calls)}\n{body}"
        )

    def test_round0_shard_i_goes_to_worker_i(self, fake_ckpt):
        """Round 0: shard index i must reach workers[i] (primary assignment)."""
        import backend.api as api_mod

        ckpt_root = str(fake_ckpt.parent.parent.parent)
        cfg = {**_FAKE_CONFIG, "ckpt_root": ckpt_root}

        sent_pairs: list[tuple[int, int]] = []  # (shard_bytes_id, worker_rank)

        # Capture which worker gets which shard_bytes object (identity via id())
        shard_bytes_order: list[int] = []

        def _capture(worker, shard_bytes, checksum, rel_path):
            sent_pairs.append((id(shard_bytes), worker["rank"]))
            return True, "", {"shard_path": "/tmp/x"}

        import backend.api as api_mod

        with patch.object(api_mod, "_load_config", return_value=cfg), \
             patch.object(api_mod, "_send_shard_to_worker", side_effect=_capture):
            client = TestClient(api_mod.app)
            client.post("/store-shard", params={"ckpt_path": str(fake_ckpt)})

        # Group by shard_bytes identity → which ranks received it
        from collections import defaultdict
        ranks_per_shard: dict[int, set[int]] = defaultdict(set)
        for shard_id, rank in sent_pairs:
            ranks_per_shard[shard_id].add(rank)

        # Every unique shard must be sent to exactly 2 distinct workers
        for shard_id, ranks in ranks_per_shard.items():
            assert len(ranks) == 2, (
                f"Shard {shard_id} was sent to {ranks} — expected exactly 2 workers"
            )

    def test_replica_is_adjacent_worker(self, fake_ckpt):
        """Round 1 must send shard i to workers[(i+1) % N], not the same as round 0."""
        import backend.api as api_mod
        from collections import defaultdict

        ckpt_root = str(fake_ckpt.parent.parent.parent)
        cfg = {**_FAKE_CONFIG, "ckpt_root": ckpt_root}

        ranks_per_shard: dict[int, list[int]] = defaultdict(list)

        def _capture(worker, shard_bytes, checksum, rel_path):
            ranks_per_shard[id(shard_bytes)].append(worker["rank"])
            return True, "", {"shard_path": "/tmp/x"}

        with patch.object(api_mod, "_load_config", return_value=cfg), \
             patch.object(api_mod, "_send_shard_to_worker", side_effect=_capture):
            client = TestClient(api_mod.app)
            client.post("/store-shard", params={"ckpt_path": str(fake_ckpt)})

        worker_ranks = [w["rank"] for w in _FAKE_WORKERS]

        for shard_id, ranks in ranks_per_shard.items():
            assert len(ranks) == 2, f"Expected 2 sends per shard, got {ranks}"
            r0, r1 = ranks[0], ranks[1]
            idx0 = worker_ranks.index(r0)
            idx1 = worker_ranks.index(r1)
            # One must be the circular successor of the other
            adjacent = (
                (idx1 == (idx0 + 1) % N_WORKERS) or
                (idx0 == (idx1 + 1) % N_WORKERS)
            )
            assert adjacent, (
                f"Ranks {r0} and {r1} are not adjacent in the worker ring {worker_ranks}"
            )

    # --- response body content ---

    def test_body_contains_round0_and_round1(self, fake_ckpt):
        body, _ = self._run_store(fake_ckpt)
        assert "[round 0]" in body, f"Missing '[round 0]' in:\n{body}"
        assert "[round 1]" in body, f"Missing '[round 1]' in:\n{body}"

    def test_done_line_reports_correct_send_count(self, fake_ckpt):
        body, _ = self._run_store(fake_ckpt)
        expected = f"{N_WORKERS * 2}/{N_WORKERS * 2} sends"
        assert expected in body, (
            f"Expected '{expected}' in body:\n{body}"
        )

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
        """If one send fails, the body must mention retry and not permanently fail."""
        call_count = {"n": 0}

        def _flaky(worker, shard_bytes, checksum, rel_path):
            call_count["n"] += 1
            # First call for rank 1 fails, all others pass
            if worker["rank"] == _FAKE_WORKERS[0]["rank"] and call_count["n"] == 1:
                return False, "connection refused", {}
            return True, "", {"shard_path": "/tmp/x"}

        body, calls = self._run_store(fake_ckpt, send_side_effect=_flaky)
        assert "queuing retry" in body, f"Expected retry message:\n{body}"
        # Despite the one flaky first attempt, the store should ultimately succeed
        assert "permanently failed" not in body, f"Should have recovered:\n{body}"

    def test_no_error_in_body_on_all_success(self, fake_ckpt):
        body, _ = self._run_store(fake_ckpt)
        assert not body.startswith("ERROR"), f"Unexpected ERROR:\n{body}"
        assert "ERROR" not in body
