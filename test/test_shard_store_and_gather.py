"""Tests for main.py — model_id conversion, remote shard counting, gather_shards HTTP.

Markers:
  (default)  — pure unit tests, no network, no SSH, always fast
  ssh        — real Fabric SSH to Pi workers from configs/config.yaml; requires cluster
  api        — real HTTP to FastAPI server; requires server running
"""
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml

from main import _count_remote_shards, gather_shards, API_BASE, REMOTE_SHARDS_ROOT
from utils.common_utils import model_id_to_dir_name

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _load_workers() -> list[dict]:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)["devices_config"]["workers"]


# ---------------------------------------------------------------------------
# model_id_to_dir_name  (pure unit)
# ---------------------------------------------------------------------------

class TestModelIdToDirName:
    def test_slash_replaced_by_double_dash(self):
        assert model_id_to_dir_name("mlx-community/Qwen2.5-0.5B-Instruct-bf16") == \
            "mlx-community--Qwen2.5-0.5B-Instruct-bf16"

    def test_no_slash_unchanged(self):
        assert model_id_to_dir_name("SmolLM2-135M") == "SmolLM2-135M"

    def test_multiple_slashes(self):
        # org/namespace/model → org--namespace--model
        assert model_id_to_dir_name("a/b/c") == "a--b--c"

    def test_already_double_dash_style_unchanged(self):
        assert model_id_to_dir_name("mlx-community--SmolLM2-135M-Instruct") == \
            "mlx-community--SmolLM2-135M-Instruct"

    def test_empty_string(self):
        assert model_id_to_dir_name("") == ""


# ---------------------------------------------------------------------------
# _count_remote_shards — real SSH to workers from config  (ssh marker)
# ---------------------------------------------------------------------------

@pytest.mark.ssh
class TestCountRemoteShardsSSH:
    """Integration tests that SSH into the actual Pi workers defined in config.yaml.
    Run with:  pytest -m ssh
    """

    @pytest.fixture(scope="class")
    def workers(self):
        return _load_workers()

    @pytest.fixture(scope="class")
    def known_model(self):
        """A model name we know is present on all workers from prior launch.sh runs."""
        with _CONFIG_PATH.open() as f:
            cfg = yaml.safe_load(f)
        return Path(cfg["data_path"]).parent.name  # e.g. mlx-community--SmolLM2-135M-Instruct

    def test_known_model_found_on_all_workers(self, workers, known_model):
        total, results = _count_remote_shards(known_model, workers)
        for w in results:
            assert w["found"] >= 1, (
                f"rank {w['rank']} ({w['host']} @ {w['ip']}) reported 0 shards "
                f"for {known_model} — was launch.sh run first?"
            )
        assert total == len(workers)

    def test_unknown_model_returns_zero(self, workers):
        total, results = _count_remote_shards("mlx-community--DoesNotExistModel", workers)
        assert total == 0
        assert all(w["found"] == 0 for w in results)

    def test_result_has_expected_keys(self, workers, known_model):
        _, results = _count_remote_shards(known_model, workers)
        for w in results:
            assert set(w.keys()) == {"rank", "host", "ip", "found"}

    def test_rank_and_ip_match_config(self, workers, known_model):
        _, results = _count_remote_shards(known_model, workers)
        for cfg_w, result_w in zip(workers, results):
            assert result_w["rank"] == cfg_w["rank"]
            assert result_w["ip"] == cfg_w["ip"]

    def test_correct_remote_path_checked(self, workers, known_model):
        """The ssh command must target .../incoming_shards/{model}/worker-{rank}/."""
        captured = []
        first_worker = workers[:1]

        original_run = subprocess.run

        def capturing_run(args, **kwargs):
            captured.append(args)
            return original_run(args, **kwargs)

        with patch("main.subprocess.run", side_effect=capturing_run):
            _count_remote_shards(known_model, first_worker)

        assert captured, "No command was run"
        full_cmd = " ".join(captured[0])
        assert known_model in full_cmd
        assert f"worker-{first_worker[0]['rank']}" in full_cmd
        assert "*.safetensors" in full_cmd


# ---------------------------------------------------------------------------
# gather_shards  (real HTTP — requires server running at API_BASE)
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestGatherShards:
    """End-to-end tests against the real FastAPI server.
    Requires: server running (bash scripts/launch.sh) + shards present on workers.
    Run with:  pytest -m api
    """

    @pytest.fixture(scope="class")
    def model_id(self) -> str:
        """HF model ID derived from the config data_path (known to exist on workers)."""
        with _CONFIG_PATH.open() as f:
            cfg = yaml.safe_load(f)
        dir_name = Path(cfg["data_path"]).parent.name
        # reverse mlx-community--SmolLM2-135M-Instruct → mlx-community/SmolLM2-135M-Instruct
        return dir_name.replace("--", "/", 1)

    def test_success_returns_body(self, model_id):
        result = gather_shards(model_id)
        assert "save_path" in result
        assert "gathered" in result

    def test_gathered_list_nonempty(self, model_id):
        result = gather_shards(model_id)
        assert len(result["gathered"]) > 0, (
            f"No shards gathered for {model_id} — are the workers running with shards present?"
        )

    def test_unknown_model_raises_http_error(self):
        with pytest.raises(httpx.HTTPStatusError):
            gather_shards("mlx-community/DoesNotExistModel-xyz")


# ---------------------------------------------------------------------------
# main() CLI behaviour  (argparse + mocks)
# ---------------------------------------------------------------------------

class TestMainCLI:
    def _run_main(self, argv: list[str], count_return=(0, []), gather_return=None):
        """Run main() with patched sys.argv, SSH counter, and gather call."""
        import main as m

        # Build workers from the real config shape so the test stays in sync
        with _CONFIG_PATH.open() as f:
            cfg = yaml.safe_load(f)
        fake_workers = cfg["devices_config"]["workers"]
        fake_config = {
            "num_workers": len(fake_workers),
            "devices_config": {"workers": fake_workers},
            "data_path": cfg["data_path"],
        }

        with patch("sys.argv", ["main.py"] + argv), \
             patch.object(m, "_load_config", return_value=fake_config), \
             patch.object(m, "_count_remote_shards", return_value=count_return) as mock_count, \
             patch.object(m, "gather_shards", return_value=gather_return or {"gathered": [], "save_path": ""}) as mock_gather:
            m.main()
            return mock_count, mock_gather

    def test_missing_model_id_exits(self):
        import main as m
        with patch("sys.argv", ["main.py"]):
            with pytest.raises(SystemExit) as exc_info:
                m.main()
        assert exc_info.value.code != 0

    def test_zero_shards_skips_gather(self):
        workers = _load_workers()
        per_worker = [{"rank": w["rank"], "host": w.get("host") or w.get("device"), "ip": w["ip"], "found": 0} for w in workers]
        mock_count, mock_gather = self._run_main(
            ["--model-id", "mlx-community/DoesNotExist"],
            count_return=(0, per_worker),
        )
        mock_gather.assert_not_called()

    def test_partial_shards_skips_gather(self):
        workers = _load_workers()
        per_worker = [{"rank": w["rank"], "host": w.get("host") or w.get("device"), "ip": w["ip"], "found": i % 2} for i, w in enumerate(workers)]
        partial_found = sum(e["found"] for e in per_worker)
        mock_count, mock_gather = self._run_main(
            ["--model-id", "mlx-community/PartialModel"],
            count_return=(partial_found, per_worker),
        )
        mock_gather.assert_not_called()

    def test_all_shards_calls_gather(self):
        workers = _load_workers()
        per_worker = [{"rank": w["rank"], "host": w.get("host") or w.get("device"), "ip": w["ip"], "found": 1} for w in workers]
        gather_return = {
            "gathered": [{"rank": w["rank"], "host": w.get("host") or w.get("device"), "shard_path": f"/tmp/shard_{w['rank']}.safetensors"} for w in workers],
            "save_path": "/tmp/out.safetensors",
        }
        mock_count, mock_gather = self._run_main(
            ["--model-id", "mlx-community/SmolLM2-135M-Instruct"],
            count_return=(len(workers), per_worker),
            gather_return=gather_return,
        )
        mock_gather.assert_called_once_with(model_id="mlx-community/SmolLM2-135M-Instruct")

    def test_model_id_converted_for_count(self):
        """model_id_to_dir_name is applied before _count_remote_shards is called."""
        workers = _load_workers()
        per_worker = [{"rank": w["rank"], "host": w.get("host") or w.get("device"), "ip": w["ip"], "found": 0} for w in workers]
        mock_count, _ = self._run_main(
            ["--model-id", "mlx-community/Qwen2.5-0.5B"],
            count_return=(0, per_worker),
        )
        name_arg = mock_count.call_args[0][0]
        assert name_arg == "mlx-community--Qwen2.5-0.5B"


# ---------------------------------------------------------------------------
# store_shard  (real HTTP — requires server running at API_BASE)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SHARDS_ROOT = _PROJECT_ROOT / "shards" / "incoming_shards"
# Ranks in the 90s are unused by real workers (config uses 1-4) so they
# won't collide with live data and are easy to clean up after the suite.
_TEST_RANKS = (96, 97, 98, 99)


@pytest.mark.api
class TestStoreShard:
    """End-to-end tests for POST /store-shard against the real FastAPI server.

    Requires: uvicorn backend.api:app running on localhost:8000.
    Run with:  pytest -m api
    """

    @pytest.fixture(scope="class")
    def config(self):
        with _CONFIG_PATH.open() as f:
            return yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def model_dir_name(self, config) -> str:
        """Directory name derived from config data_path, e.g. mlx-community--SmolLM2-135M-Instruct."""
        return Path(config["data_path"]).parent.name

    @pytest.fixture(scope="class")
    def model_id(self, model_dir_name) -> str:
        """HF model ID, e.g. mlx-community/SmolLM2-135M-Instruct."""
        return model_dir_name.replace("--", "/", 1)

    @pytest.fixture(scope="class")
    def fixture_shard(self, config) -> bytes:
        """Raw bytes of the real safetensors file pointed to by config data_path."""
        return (_PROJECT_ROOT / config["data_path"]).read_bytes()

    @pytest.fixture(autouse=True, scope="class")
    def _cleanup(self, model_dir_name):
        yield
        for rank in _TEST_RANKS:
            d = _SHARDS_ROOT / model_dir_name / f"worker-{rank}"
            shutil.rmtree(d, ignore_errors=True)

    def _post(self, rank: int, shard_bytes: bytes, **form_fields) -> httpx.Response:
        with httpx.Client() as client:
            return client.post(
                f"{API_BASE}/store-shard",
                data={"rank": rank, **form_fields},
                files={"file": ("model.safetensors", shard_bytes, "application/octet-stream")},
                timeout=60.0,
            )

    def test_with_model_id_writes_to_correct_path(self, fixture_shard, model_id, model_dir_name):
        resp = self._post(99, fixture_shard, model_id=model_id)
        resp.raise_for_status()
        expected_dir = _SHARDS_ROOT / model_dir_name / "worker-99"
        assert Path(resp.json()["shard_path"]).parent == expected_dir

    def test_response_has_required_keys(self, fixture_shard, model_id):
        resp = self._post(98, fixture_shard, model_id=model_id)
        resp.raise_for_status()
        assert {"shard_path", "metadata_path", "rank"} <= resp.json().keys()

    def test_rank_in_response_matches_sent_rank(self, fixture_shard, model_id):
        resp = self._post(97, fixture_shard, model_id=model_id)
        resp.raise_for_status()
        assert resp.json()["rank"] == 97

    def test_without_model_id_falls_back_to_config_path(self, fixture_shard, model_dir_name):
        resp = self._post(96, fixture_shard)  # no model_id — server reads config
        resp.raise_for_status()
        expected_dir = _SHARDS_ROOT / model_dir_name / "worker-96"
        assert Path(resp.json()["shard_path"]).parent == expected_dir

    def test_missing_rank_returns_422(self, fixture_shard):
        with httpx.Client() as client:
            resp = client.post(
                f"{API_BASE}/store-shard",
                files={"file": ("model.safetensors", fixture_shard, "application/octet-stream")},
                timeout=30.0,
            )
        assert resp.status_code == 422
