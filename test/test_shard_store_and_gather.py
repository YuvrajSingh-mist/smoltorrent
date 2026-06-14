"""Tests for main.py — model_id conversion, remote shard counting, gather_shards HTTP.

Markers:
  (default)  — pure unit tests, no network, no SSH, always fast
  ssh        — real Fabric SSH to Pi workers from configs/config.yaml; requires cluster
  api        — real HTTP to FastAPI server; requires server running
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml

from utils.check_workers import count_remote_shards
from utils.common_utils import gather_shards, API_BASE, model_id_to_dir_name

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _load_workers() -> list[dict]:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)["devices_config"]["workers"]


# ---------------------------------------------------------------------------
# model_id_to_dir_name  (pure unit)
# ---------------------------------------------------------------------------


class TestModelIdToDirName:
    def test_slash_replaced_by_double_dash(self):
        assert (
            model_id_to_dir_name("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
            == "mlx-community--Qwen2.5-0.5B-Instruct-bf16"
        )

    def test_no_slash_unchanged(self):
        assert model_id_to_dir_name("SmolLM2-135M") == "SmolLM2-135M"

    def test_multiple_slashes(self):
        # org/namespace/model → org--namespace--model
        assert model_id_to_dir_name("a/b/c") == "a--b--c"

    def test_already_double_dash_style_unchanged(self):
        assert (
            model_id_to_dir_name("mlx-community--SmolLM2-135M-Instruct")
            == "mlx-community--SmolLM2-135M-Instruct"
        )

    def test_empty_string(self):
        assert model_id_to_dir_name("") == ""


# ---------------------------------------------------------------------------
# count_remote_shards — real SSH to workers from config  (ssh marker)
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
        return Path(
            cfg["data_path"]
        ).parent.name  # e.g. mlx-community--SmolLM2-135M-Instruct

    def test_known_model_found_on_all_workers(self, workers, known_model):
        total, results = count_remote_shards(known_model, workers)
        for w in results:
            assert w["found"] >= 1, (
                f"rank {w['rank']} ({w['host']} @ {w['ip']}) reported 0 shards "
                f"for {known_model} — was launch.sh run first?"
            )
        assert total == len(workers)

    def test_unknown_model_returns_zero(self, workers):
        total, results = count_remote_shards(
            "mlx-community--DoesNotExistModel", workers
        )
        assert total == 0
        assert all(w["found"] == 0 for w in results)

    def test_result_has_expected_keys(self, workers, known_model):
        _, results = count_remote_shards(known_model, workers)
        for w in results:
            assert set(w.keys()) == {"rank", "host", "ip", "found"}

    def test_rank_and_ip_match_config(self, workers, known_model):
        _, results = count_remote_shards(known_model, workers)
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

        with patch("utils.check_workers.subprocess.run", side_effect=capturing_run):
            count_remote_shards(known_model, first_worker)

        assert captured, "No command was run"
        full_cmd = " ".join(captured[0])
        assert known_model in full_cmd
        assert f"worker_{first_worker[0]['rank']}" in full_cmd
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
# store_shard  (real HTTP — requires server running at API_BASE)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.api
class TestStoreShard:
    """POST /store-shard?ckpt_path=<path>: shard a checkpoint and push to Pi workers.

    Requires: uvicorn backend.api:app running on localhost:8000 + workers reachable.
    Run with:  pytest -m api
    """

    @pytest.fixture(scope="class")
    def config(self):
        with _CONFIG_PATH.open() as f:
            return yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def ckpt_path(self, config) -> str:
        """First real safetensors file found under ckpt_root."""
        root = Path(config["ckpt_root"]).expanduser()
        candidates = sorted(root.rglob("model.safetensors"))
        if not candidates:
            pytest.skip(f"No model.safetensors found under {root}")
        return str(candidates[0])

    def _stream_body(self, ckpt_path: str) -> str:
        with httpx.Client(timeout=None) as client:
            with client.stream(
                "POST", f"{API_BASE}/store-shard", params={"ckpt_path": ckpt_path}
            ) as resp:
                resp.raise_for_status()
                return resp.read().decode()

    def test_store_succeeds_no_error(self, ckpt_path):
        body = self._stream_body(ckpt_path)
        assert "ERROR" not in body, f"Store failed:\n{body}"

    def test_store_done_line_present(self, ckpt_path):
        body = self._stream_body(ckpt_path)
        assert "Done:" in body

    def test_store_all_workers_acked(self, ckpt_path, config):
        body = self._stream_body(ckpt_path)
        for w in config["devices_config"]["workers"]:
            assert f"rank {w['rank']}" in body

    def test_missing_ckpt_path_returns_error(self):
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{API_BASE}/store-shard")
        assert resp.status_code == 422
