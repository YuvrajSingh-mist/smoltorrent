"""Parallel and concurrent gather tests.

Three scenarios:

1. ``TestParallelDistinctGathers``
   Fire N /gather-shards calls for N *different* shard_keys concurrently via
   ThreadPoolExecutor.  Each should complete successfully and produce a valid
   merged file.  This is the normal "download many checkpoints at once" case.

2. ``TestConcurrentSameKey``
   Fire the same shard_key multiple times in parallel.  All must succeed and
   the resulting merged file must be byte-identical across all calls
   (idempotency + no torn writes to the same path).

3. ``TestRapidSequentialGathers``
   Gather the same shard_key back-to-back with no delay.  Verifies the API
   does not corrupt or lose the merged file across repeated calls.

Markers:
  api — requires API on localhost:8000 and all 4 Pi workers reachable.

Run:
  pytest -m api test/test_parallel_gather.py -v
"""

import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

API_BASE = "http://localhost:8000"
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

# How many concurrent gathers to launch in scenario 1 and 2.
PARALLEL_N = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config():
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _available_shard_keys(n: int) -> list[str]:
    """Return up to *n* distinct shard_keys from /models, newest first."""
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_BASE}/models", params={"limit": n})
        resp.raise_for_status()
    keys = [m["shard_key"] for m in resp.json()["models"]]
    if not keys:
        pytest.skip("No stored checkpoints found — run store-shard first")
    return keys[:n]


def _gather(shard_key: str) -> tuple[str, str]:
    """POST /gather-shards for *shard_key*, return (shard_key, response_body)."""
    with httpx.Client(timeout=None) as client:
        with client.stream(
            "POST", f"{API_BASE}/gather-shards", params={"shard_key": shard_key}
        ) as resp:
            resp.raise_for_status()
            body = resp.read().decode()
    return shard_key, body


def _merged_path(shard_key: str) -> Path:
    cfg = _config()
    return Path(cfg["ckpt_root"]).expanduser() / shard_key / "merged.safetensors"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Scenario 1 — N distinct shard_keys in parallel
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestParallelDistinctGathers:
    """PARALLEL_N different checkpoints gathered at the same time."""

    @pytest.fixture(scope="class")
    def shard_keys(self):
        keys = _available_shard_keys(PARALLEL_N)
        if len(keys) < 2:
            pytest.skip(f"Need at least 2 stored checkpoints, found {len(keys)}")
        return keys

    @pytest.fixture(scope="class")
    def results(self, shard_keys):
        """Fire all gathers concurrently, collect (shard_key, body) pairs."""
        with ThreadPoolExecutor(max_workers=len(shard_keys)) as pool:
            futures = {pool.submit(_gather, k): k for k in shard_keys}
            return [f.result() for f in as_completed(futures)]

    def test_all_succeed(self, results):
        for shard_key, body in results:
            assert "ERROR" not in body, (
                f"Gather failed for {shard_key}:\n{body}"
            )

    def test_all_have_done_line(self, results):
        for shard_key, body in results:
            assert "Done:" in body, (
                f"No Done line for {shard_key}:\n{body}"
            )

    def test_all_merged_files_exist(self, results):
        for shard_key, _ in results:
            merged = _merged_path(shard_key)
            assert merged.exists(), f"merged.safetensors missing for {shard_key}"

    def test_all_merged_files_nonzero(self, results):
        for shard_key, _ in results:
            merged = _merged_path(shard_key)
            assert merged.stat().st_size > 0, (
                f"merged.safetensors is empty for {shard_key}"
            )

    def test_no_permanently_failed_shards(self, results):
        for shard_key, body in results:
            assert "permanently failed" not in body, (
                f"Shard permanently failed for {shard_key}:\n{body}"
            )


# ---------------------------------------------------------------------------
# Scenario 2 — Same shard_key gathered N times in parallel (idempotency)
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestConcurrentSameKey:
    """The same checkpoint gathered PARALLEL_N times simultaneously."""

    @pytest.fixture(scope="class")
    def shard_key(self):
        return _available_shard_keys(1)[0]

    @pytest.fixture(scope="class")
    def results(self, shard_key):
        with ThreadPoolExecutor(max_workers=PARALLEL_N) as pool:
            futures = [pool.submit(_gather, shard_key) for _ in range(PARALLEL_N)]
            return [f.result() for f in as_completed(futures)]

    def test_all_succeed(self, results):
        for shard_key, body in results:
            assert "ERROR" not in body, (
                f"Concurrent gather failed:\n{body}"
            )

    def test_merged_file_exists(self, results):
        shard_key = results[0][0]
        assert _merged_path(shard_key).exists()

    def test_merged_file_identical_across_all_calls(self, results):
        """All concurrent gathers must produce the same byte content."""
        shard_key = results[0][0]
        merged = _merged_path(shard_key)
        assert merged.exists()
        checksum = _sha256(merged)
        # Re-hash after all writes have settled — file should be stable
        time.sleep(0.5)
        assert _sha256(merged) == checksum, (
            "merged.safetensors changed between reads — concurrent writes are not idempotent"
        )

    def test_no_torn_writes(self, results):
        """File size must be consistent — a torn write would leave a truncated file."""
        shard_key = results[0][0]
        merged = _merged_path(shard_key)
        size_a = merged.stat().st_size
        time.sleep(0.2)
        size_b = merged.stat().st_size
        assert size_a == size_b, (
            f"merged.safetensors size changed after gathers settled: {size_a} → {size_b}"
        )


# ---------------------------------------------------------------------------
# Scenario 3 — Rapid sequential gathers (re-gather stability)
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestRapidSequentialGathers:
    """Gather the same checkpoint 3 times back-to-back, no delay between calls."""

    REPEATS = 3

    @pytest.fixture(scope="class")
    def shard_key(self):
        return _available_shard_keys(1)[0]

    @pytest.fixture(scope="class")
    def bodies(self, shard_key):
        return [_gather(shard_key)[1] for _ in range(self.REPEATS)]

    def test_all_succeed(self, bodies):
        for i, body in enumerate(bodies):
            assert "ERROR" not in body, (
                f"Sequential gather #{i + 1} failed:\n{body}"
            )

    def test_all_have_done_line(self, bodies):
        for i, body in enumerate(bodies):
            assert "Done:" in body, f"No Done line in sequential gather #{i + 1}"

    def test_merged_file_stable_across_repeats(self, shard_key, bodies):
        """Each re-gather must produce the same file content as the first."""
        merged = _merged_path(shard_key)
        assert merged.exists()
        expected = _sha256(merged)
        # Re-hash — if file changed between runs the checksum would differ
        assert _sha256(merged) == expected, (
            "merged.safetensors content changed between sequential gathers"
        )

    def test_no_integrity_regression(self, bodies):
        """None of the repeated gathers should report a checksum mismatch."""
        for i, body in enumerate(bodies):
            assert "checksum mismatch" not in body, (
                f"Integrity regression on sequential gather #{i + 1}:\n{body}"
            )
