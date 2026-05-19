"""Tests for store-shard and gather-shards with REDUNDANCY=2.

Markers:
  api           — requires API running on localhost:8000 + all 4 Pi workers up.
  ssh           — replica-fallback test: kills pi4-1 over SSH, gathers, restores.

Run:
  pytest -m api                         # store + gather + discover
  pytest -m "api or ssh"                # + replica fallback
"""
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

API_BASE = "http://localhost:8000"
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
def _config():
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _ckpt_path() -> str:
    """Return a real checkpoint path that exists under ckpt_root."""
    cfg = _config()
    root = Path(cfg["ckpt_root"]).expanduser()
    candidates = sorted(root.rglob("model.safetensors"))
    if not candidates:
        pytest.skip(f"No model.safetensors found under {root}")
    return str(candidates[0])


def _stream_post(endpoint: str, ckpt_path: str) -> str:
    """POST to a streaming endpoint, collect and return full body as string."""
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", f"{API_BASE}/{endpoint}", params={"ckpt_path": ckpt_path}) as resp:
            resp.raise_for_status()
            return resp.read().decode()


# ---------------------------------------------------------------------------
# /store-shard with REDUNDANCY=2
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestStoreShard:

    @pytest.fixture(scope="class")
    def ckpt_path(self):
        return _ckpt_path()

    @pytest.fixture(scope="class")
    def store_body(self, ckpt_path):
        return _stream_post("store-shard", ckpt_path)

    def test_store_succeeds(self, store_body):
        assert "ERROR" not in store_body, f"Store failed:\n{store_body}"

    def test_store_done_line_present(self, store_body):
        assert "Done:" in store_body, f"No Done line in store output:\n{store_body}"

    def test_store_sends_two_rounds(self, store_body):
        """With REDUNDANCY=2 every rank should appear twice (round 0 + round 1)."""
        cfg = _config()
        num_workers = len(cfg["devices_config"]["workers"])
        for w in cfg["devices_config"]["workers"]:
            rank = w["rank"]
            count = store_body.count(f"rank {rank}")
            assert count >= 2, (
                f"rank {rank} appears only {count} time(s) — expected 2 rounds"
            )

    def test_store_reports_correct_total_sends(self, store_body):
        """Done line should say 'N*2/N*2 sends (2x replicated)'."""
        cfg = _config()
        n = len(cfg["devices_config"]["workers"])
        expected = f"{n * 2}/{n * 2} sends"
        assert expected in store_body, (
            f"Expected '{expected}' in store output:\n{store_body}"
        )

    def test_store_no_permanent_failures(self, store_body):
        assert "permanently failed" not in store_body

    def test_store_round0_and_round1_logged(self, store_body):
        assert "[round 0]" in store_body, "round 0 not logged"
        assert "[round 1]" in store_body, "round 1 not logged"


# ---------------------------------------------------------------------------
# /gather-shards
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestGatherShards:

    @pytest.fixture(scope="class")
    def ckpt_path(self):
        return _ckpt_path()

    @pytest.fixture(scope="class")
    def gather_body(self, ckpt_path):
        # Ensure shards are stored first
        _stream_post("store-shard", ckpt_path)
        return _stream_post("gather-shards", ckpt_path)

    def test_gather_succeeds(self, gather_body):
        assert "ERROR" not in gather_body, f"Gather failed:\n{gather_body}"

    def test_gather_done_line_present(self, gather_body):
        assert "Done: saved →" in gather_body

    def test_gather_all_shards_collected(self, gather_body):
        cfg = _config()
        for i in range(len(cfg["devices_config"]["workers"])):
            assert f"shard {i}" in gather_body, f"shard {i} not mentioned in gather output"

    def test_merged_file_exists(self, ckpt_path, gather_body):
        cfg = _config()
        ckpt_root = Path(cfg["ckpt_root"]).expanduser()
        ckpt_file = Path(ckpt_path)
        rel = ckpt_file.parent.relative_to(ckpt_root)
        merged = ckpt_root / rel / "merged.safetensors"
        assert merged.exists(), f"merged.safetensors not found at {merged}"

    def test_gather_no_failed_shards(self, gather_body):
        assert "skipping merge" not in gather_body


# ---------------------------------------------------------------------------
# Replica fallback (requires SSH to kill/restart a Pi worker)
# ---------------------------------------------------------------------------

@pytest.mark.api
@pytest.mark.ssh
class TestReplicaFallback:
    """Kill pi4-1's worker, gather — should fall back to pi4-2 for shard 0."""

    _PI = "pi4-1"
    _RANK = 1
    _REPLICA_RANK = 2

    @pytest.fixture(scope="class")
    def ckpt_path(self):
        return _ckpt_path()

    @pytest.fixture(scope="class", autouse=True)
    def ensure_stored(self, ckpt_path):
        """Store shards (both rounds) before any fallback test."""
        _stream_post("store-shard", ckpt_path)

    @pytest.fixture(scope="class")
    def fallback_body(self, ckpt_path):
        """Kill pi4-1 once, gather once, restore — all three tests share this result."""
        subprocess.run(["ssh", self._PI, "pkill -f 'worker.py'"], capture_output=True)
        time.sleep(3)  # wait for port to close
        body = _stream_post("gather-shards", ckpt_path)
        # Restore pi4-1
        subprocess.run([
            "ssh", self._PI,
            f"cd ~/Desktop/smoltorrent && tmux new-session -d -s syncps_worker_{self._RANK} "
            f"\"bash -lc '.venv/bin/python algorithms/SyncPS/worker.py {self._RANK} {self._PI} "
            f"2>&1 | tee /tmp/smolcluster-logs/syncps-worker-rank{self._RANK}-{self._PI}.log; exec bash'\""
        ], capture_output=True)
        time.sleep(6)  # wait for worker to fully bind + re-advertise
        return body

    def test_gather_succeeds_with_primary_down(self, fallback_body):
        assert "ERROR" not in fallback_body, f"Gather failed with pi4-1 down:\n{fallback_body}"
        assert "Done: saved →" in fallback_body

    def test_fallback_message_logged(self, fallback_body):
        assert f"trying replica rank {self._REPLICA_RANK}" in fallback_body, (
            f"Expected replica fallback log not found:\n{fallback_body}"
        )

    def test_merged_file_still_written(self, ckpt_path, fallback_body):
        cfg = _config()
        ckpt_root = Path(cfg["ckpt_root"]).expanduser()
        rel = Path(ckpt_path).parent.relative_to(ckpt_root)
        merged = ckpt_root / rel / "merged.safetensors"
        assert merged.exists()
