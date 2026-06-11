"""Tests for main.py — CLI argument handling and remote shard counting.

Markers:
  (default) — pure unit tests, no network, always fast
  ssh       — real SSH to Pi workers from configs/config.yaml; requires cluster reachable
              Run with:  pytest -m ssh
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml



from utils.check_workers import count_remote_shards

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _load_workers() -> list[dict]:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)["devices_config"]["workers"]


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
        with _CONFIG_PATH.open() as f:
            cfg = yaml.safe_load(f)
        return Path(cfg["data_path"]).parent.name

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
