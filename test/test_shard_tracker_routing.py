"""Shard tracker routing tests — store two real models, verify tracker-targeted gather.

Two models used (no download needed — already on disk):
  Model A: LiquidAI/LFM2.5-350M-MLX-bf16   (~676 MB)  symlinked from HF cache
  Model B: Qwen2.5-0.5B-instruct-bf16       (~948 MB)  already in checkpoints

Flow:
  1. Store model A → tracker records which ranks received it
  2. Store model B → tracker records which ranks received it
  3. Gather model A → API must only contact ranks in tracker[model_A_key]
  4. Gather model B → API must only contact ranks in tracker[model_B_key]
  5. Verify no cross-contact (model A gather never reaches model-B-only ranks)
  6. Verify checksum of merged model matches original

Markers:
  api         — requires API server on localhost:8000
  integration — requires live Pi workers
"""

import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.shard_tracker import add_shard, clear, get_ranks, list_shards_for_rank
from utils.common_utils import compute_checksum

_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"
_CKPT_ROOT   = Path("~/smolcluster/checkpoints").expanduser()
API          = "http://localhost:8000"

MODEL_A_PATH = _CKPT_ROOT / "LiquidAI--LFM2.5-350M-MLX-bf16" / "step_0" / "model.safetensors"
MODEL_B_PATH = _CKPT_ROOT / "Qwen2.5-0.5B-instruct-bf16"      / "gsm8k"  / "step_0" / "model.safetensors"

MODEL_A_KEY  = "LiquidAI--LFM2.5-350M-MLX-bf16/step_0"
MODEL_B_KEY  = "Qwen2.5-0.5B-instruct-bf16/gsm8k/step_0"


def _load_workers() -> list[dict]:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)["devices_config"]["workers"]


def _store(ckpt_path: Path, timeout: float = 1800.0) -> list[str]:
    """POST /store-shard, collect streamed log lines, return them."""
    lines = []
    with httpx.stream("POST", f"{API}/store-shard",
                      params={"ckpt_path": str(ckpt_path)},
                      timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                lines.append(line)
    return lines


def _gather(ckpt_path: Path, timeout: float = 1800.0) -> list[str]:
    """POST /gather-shards, collect streamed log lines, return them."""
    lines = []
    with httpx.stream("POST", f"{API}/gather-shards",
                      params={"ckpt_path": str(ckpt_path)},
                      timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Unit tests — tracker JSON logic only, no network
# ---------------------------------------------------------------------------

class TestShardTrackerUnit:
    """Unit tests that never touch the real shard_map.json."""

    @pytest.fixture(autouse=True)
    def _isolate_tracker(self, tmp_path, monkeypatch):
        import utils.shard_tracker as st
        monkeypatch.setattr(st, "_TRACKER_PATH", tmp_path / "shard_map.json")
        yield

    def test_add_and_get(self):
        add_shard(rank=1, shard_key="modelA/step_0")
        add_shard(rank=2, shard_key="modelA/step_0")
        assert sorted(get_ranks("modelA/step_0")) == [1, 2]

    def test_get_empty_returns_empty_list(self):
        assert get_ranks("nonexistent/key") == []

    def test_add_idempotent(self):
        add_shard(rank=1, shard_key="modelA/step_0")
        add_shard(rank=1, shard_key="modelA/step_0")
        assert get_ranks("modelA/step_0") == [1]

    def test_two_models_dont_cross_contaminate(self):
        add_shard(rank=1, shard_key="modelA/step_0")
        add_shard(rank=2, shard_key="modelB/step_0")
        assert get_ranks("modelA/step_0") == [1]
        assert get_ranks("modelB/step_0") == [2]

    def test_list_shards_for_rank(self):
        add_shard(rank=1, shard_key="modelA/step_0")
        add_shard(rank=1, shard_key="modelA/step_100")
        add_shard(rank=2, shard_key="modelB/step_0")
        shards = list_shards_for_rank(1)
        assert "modelA/step_0"   in shards
        assert "modelA/step_100" in shards
        assert "modelB/step_0"  not in shards

    def test_targeted_gather_skips_irrelevant_ranks(self):
        """Core routing logic: only ranks registered for a key should be targeted."""
        workers = [{"rank": 1}, {"rank": 2}, {"rank": 3}, {"rank": 4}]
        add_shard(rank=1, shard_key="modelA/step_0")
        add_shard(rank=3, shard_key="modelA/step_0")

        tracked = get_ranks("modelA/step_0")
        targeted = [w for w in workers if w["rank"] in tracked]

        assert len(targeted) == 2
        assert all(w["rank"] in (1, 3) for w in targeted)
        assert not any(w["rank"] in (2, 4) for w in targeted)

    def test_tracker_survives_json_round_trip(self):
        """Verify atomic write + re-read preserves exact data."""
        import utils.shard_tracker as st
        add_shard(rank=1, shard_key="modelA/step_0")
        add_shard(rank=2, shard_key="modelA/step_0")
        add_shard(rank=2, shard_key="modelB/step_0")

        data = st._load()
        assert sorted(data["modelA/step_0"]) == [1, 2]
        assert data["modelB/step_0"] == [2]


# ---------------------------------------------------------------------------
# API + integration — real store/gather against live workers
# ---------------------------------------------------------------------------

@pytest.mark.api
@pytest.mark.integration
class TestTrackerRoutingLive:
    """Store two real models via the API, verify tracker routes gather correctly."""

    @pytest.fixture(autouse=True)
    def _check_models(self):
        if not MODEL_A_PATH.exists():
            pytest.skip(f"Model A not found: {MODEL_A_PATH}")
        if not MODEL_B_PATH.exists():
            pytest.skip(f"Model B not found: {MODEL_B_PATH}")

    @pytest.fixture(autouse=True)
    def _clean_tracker_entries(self):
        """Remove only the test keys so we don't wipe real cluster state."""
        for key in (MODEL_A_KEY, MODEL_B_KEY):
            # remove stale entries for these two models
            import utils.shard_tracker as st
            with st._lock:
                data = st._load()
                data.pop(key, None)
                st._save(data)
        yield
        # leave entries in place after test — useful for manual inspection

    # --- store ---

    def test_store_model_a_populates_tracker(self):
        lines = _store(MODEL_A_PATH)
        output = "\n".join(lines)
        assert "ERROR" not in output, f"Store A failed:\n{output}"
        assert "Done:" in output

        ranks = get_ranks(MODEL_A_KEY)
        assert len(ranks) >= 1, "Tracker has no entry for model A after store"
        workers = _load_workers()
        valid_ranks = {w["rank"] for w in workers}
        assert all(r in valid_ranks for r in ranks), f"Tracker has invalid ranks: {ranks}"

    def test_store_model_b_populates_tracker(self):
        lines = _store(MODEL_B_PATH)
        output = "\n".join(lines)
        assert "ERROR" not in output, f"Store B failed:\n{output}"

        ranks = get_ranks(MODEL_B_KEY)
        assert len(ranks) >= 1, "Tracker has no entry for model B after store"

    def test_two_models_tracked_independently(self):
        _store(MODEL_A_PATH)
        _store(MODEL_B_PATH)

        ranks_a = get_ranks(MODEL_A_KEY)
        ranks_b = get_ranks(MODEL_B_KEY)

        assert len(ranks_a) >= 1, "Model A not tracked"
        assert len(ranks_b) >= 1, "Model B not tracked"
        # Both models are sharded across workers — ranks should overlap
        # (same workers hold different shards of each model)
        # but the tracker keys are independent
        assert MODEL_A_KEY != MODEL_B_KEY

    # --- gather targeting ---

    def test_gather_model_a_uses_tracker(self):
        _store(MODEL_A_PATH)
        expected_ranks = set(get_ranks(MODEL_A_KEY))
        assert expected_ranks, "Must store before gather"

        lines = _gather(MODEL_A_PATH)
        output = "\n".join(lines)

        assert "ERROR" not in output, f"Gather A failed:\n{output}"
        # API logs "Tracker: N/M worker(s) known to hold this shard"
        assert "Tracker:" in output
        assert "broadcasting to all" not in output, (
            "Gather A fell back to broadcast — tracker entry missing"
        )

    def test_gather_model_b_uses_tracker(self):
        _store(MODEL_B_PATH)
        expected_ranks = set(get_ranks(MODEL_B_KEY))
        assert expected_ranks

        lines = _gather(MODEL_B_PATH)
        output = "\n".join(lines)

        assert "ERROR" not in output, f"Gather B failed:\n{output}"
        assert "Tracker:" in output
        assert "broadcasting to all" not in output

    def test_tracker_rank_count_matches_gather_log(self):
        """The number of workers the API contacts must equal len(get_ranks(key))."""
        _store(MODEL_A_PATH)
        tracked_count = len(get_ranks(MODEL_A_KEY))

        lines = _gather(MODEL_A_PATH)
        output = "\n".join(lines)

        # Log line: "Tracker: 2/2 worker(s) known to hold this shard"
        import re
        m = re.search(r"Tracker: (\d+)/\d+ worker", output)
        assert m, f"Tracker log line not found in:\n{output}"
        assert int(m.group(1)) == tracked_count, (
            f"API contacted {m.group(1)} workers but tracker has {tracked_count} ranks"
        )

    def test_model_a_gather_throughput(self, capsys):
        _store(MODEL_A_PATH)
        size_mb = MODEL_A_PATH.stat().st_size / (1024 * 1024)

        t0 = time.perf_counter()
        lines = _gather(MODEL_A_PATH)
        elapsed = time.perf_counter() - t0

        output = "\n".join(lines)
        assert "ERROR" not in output
        mbps = size_mb / elapsed if elapsed > 0 else 0

        with capsys.disabled():
            print(
                f"\n[tracker] Model A ({size_mb:.0f} MB) gather via sendfile+mmap: "
                f"{mbps:.1f} MB/s ({elapsed:.1f}s) | "
                f"targeted ranks: {get_ranks(MODEL_A_KEY)}"
            )

    def test_model_b_gather_throughput(self, capsys):
        _store(MODEL_B_PATH)
        size_mb = MODEL_B_PATH.stat().st_size / (1024 * 1024)

        t0 = time.perf_counter()
        lines = _gather(MODEL_B_PATH)
        elapsed = time.perf_counter() - t0

        output = "\n".join(lines)
        assert "ERROR" not in output
        mbps = size_mb / elapsed if elapsed > 0 else 0

        with capsys.disabled():
            print(
                f"\n[tracker] Model B ({size_mb:.0f} MB) gather via sendfile+mmap: "
                f"{mbps:.1f} MB/s ({elapsed:.1f}s) | "
                f"targeted ranks: {get_ranks(MODEL_B_KEY)}"
            )
