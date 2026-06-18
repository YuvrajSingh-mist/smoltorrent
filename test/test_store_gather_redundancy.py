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


def _store(ckpt_path: str) -> str:
    """POST /store-shard with an absolute checkpoint path, return full streamed body."""
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", f"{API_BASE}/store-shard", params={"ckpt_path": ckpt_path}) as resp:
            resp.raise_for_status()
            return resp.read().decode()


def _gather(shard_key: str) -> str:
    """POST /gather-shards with a shard_key, return full streamed body."""
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", f"{API_BASE}/gather-shards", params={"shard_key": shard_key}) as resp:
            resp.raise_for_status()
            return resp.read().decode()


def _extract_shard_key(store_body: str) -> str:
    """Parse the 'Shard key: ...' line emitted at the end of a successful store."""
    for line in store_body.splitlines():
        if line.startswith("Shard key: "):
            return line.removeprefix("Shard key: ").strip()
    raise ValueError(f"No 'Shard key:' line found in store response:\n{store_body}")


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
        return _store(ckpt_path)

    def test_store_succeeds(self, store_body):
        assert "ERROR" not in store_body, f"Store failed:\n{store_body}"

    def test_store_done_line_present(self, store_body):
        assert "Done:" in store_body, f"No Done line in store output:\n{store_body}"

    def test_store_emits_shard_key(self, store_body):
        key = _extract_shard_key(store_body)
        assert key, f"Shard key is empty:\n{store_body}"

    def test_store_sends_two_rounds(self, store_body):
        cfg = _config()
        for w in cfg["devices_config"]["workers"]:
            rank = w["rank"]
            count = store_body.count(f"rank {rank}")
            assert count >= 2, (
                f"rank {rank} appears only {count} time(s) — expected 2 rounds"
            )

    def test_store_reports_correct_total_sends(self, store_body):
        cfg = _config()
        n = len(cfg["devices_config"]["workers"])
        expected = f"{n * 2}/{n * 2} sends"
        assert expected in store_body, (
            f"Expected '{expected}' in store output:\n{store_body}"
        )

    def test_store_no_permanent_failures(self, store_body):
        assert "permanently failed" not in store_body

    def test_store_round0_and_round1_logged(self, store_body):
        assert "shard_0.safetensors" in store_body, "primary shard filename not logged"
        assert "shard_1.safetensors" in store_body, "replica shard filename not logged"


# ---------------------------------------------------------------------------
# /gather-shards — uses shard_key from store response, no ckpt_path
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestGatherShards:
    @pytest.fixture(scope="class")
    def shard_key(self):
        store_body = _store(_ckpt_path())
        return _extract_shard_key(store_body)

    @pytest.fixture(scope="class")
    def gather_body(self, shard_key):
        return _gather(shard_key)

    def test_gather_succeeds(self, gather_body):
        assert "ERROR" not in gather_body, f"Gather failed:\n{gather_body}"

    def test_gather_done_line_present(self, gather_body):
        assert "Done:" in gather_body, f"No Done line in gather output:\n{gather_body}"

    def test_gather_all_shards_collected(self, gather_body):
        cfg = _config()
        for i in range(len(cfg["devices_config"]["workers"])):
            assert f"shard {i}" in gather_body, (
                f"shard {i} not mentioned in gather output"
            )

    def test_merged_file_exists(self, shard_key, gather_body):
        cfg = _config()
        ckpt_root = Path(cfg["ckpt_root"]).expanduser()
        merged = ckpt_root / shard_key / "merged.safetensors"
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
    _RANK = 3          # minilab-pi4-1 is rank 3 in config
    _REPLICA_RANK = 4  # shard replica for rank 3 lands on rank 4 (minilab-pi4-2)

    @pytest.fixture(scope="class")
    def shard_key(self):
        store_body = _store(_ckpt_path())
        return _extract_shard_key(store_body)

    @pytest.fixture(scope="class")
    def fallback_body(self, shard_key):
        """Kill pi4-1 once, gather once, restore — all three tests share this result."""
        subprocess.run(["ssh", self._PI, "pkill -f 'worker.py'"], capture_output=True)
        time.sleep(3)
        body = _gather(shard_key)
        subprocess.run(
            [
                "ssh",
                self._PI,
                f"cd ~/Desktop/smoltorrent && tmux new-session -d -s syncps_worker_{self._RANK} "
                f"\"bash -lc '.venv/bin/python algorithms/SyncPS/worker.py {self._RANK} {self._PI} "
                f"2>&1 | tee /tmp/smolcluster-logs/syncps-worker-rank{self._RANK}-{self._PI}.log; exec bash'\"",
            ],
            capture_output=True,
        )
        time.sleep(6)
        return body

    def test_gather_succeeds_with_primary_down(self, fallback_body):
        assert "ERROR" not in fallback_body, (
            f"Gather failed with pi4-1 down:\n{fallback_body}"
        )
        assert "Done:" in fallback_body

    def test_fallback_message_logged(self, fallback_body):
        assert f"rank {self._REPLICA_RANK}" in fallback_body, (
            f"Expected replica rank {self._REPLICA_RANK} in gather output:\n{fallback_body}"
        )

    def test_merged_file_still_written(self, shard_key, fallback_body):
        cfg = _config()
        ckpt_root = Path(cfg["ckpt_root"]).expanduser()
        merged = ckpt_root / shard_key / "merged.safetensors"
        assert merged.exists()
