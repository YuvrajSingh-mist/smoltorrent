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

from main import _count_remote_shards

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _load_workers() -> list[dict]:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)["devices_config"]["workers"]


# ---------------------------------------------------------------------------
# main() CLI behaviour  (unit — no network)
# ---------------------------------------------------------------------------


class TestMainCLI:
    def _run_main(self, argv: list[str], count_return=(0, []), gather_return=None):
        """Run main() with patched sys.argv, SSH counter, and gather call."""
        import main as m

        with _CONFIG_PATH.open() as f:
            cfg = yaml.safe_load(f)
        fake_workers = cfg["devices_config"]["workers"]
        fake_config = {
            "num_workers": len(fake_workers),
            "devices_config": {"workers": fake_workers},
            "data_path": cfg["data_path"],
        }

        with (
            patch("sys.argv", ["main.py"] + argv),
            patch.object(m, "_load_config", return_value=fake_config),
            patch.object(
                m, "_count_remote_shards", return_value=count_return
            ) as mock_count,
            patch.object(
                m,
                "gather_shards",
                return_value=gather_return or {"gathered": [], "save_path": ""},
            ) as mock_gather,
        ):
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
        per_worker = [
            {
                "rank": w["rank"],
                "host": w.get("host") or w.get("device"),
                "ip": w["ip"],
                "found": 0,
            }
            for w in workers
        ]
        mock_count, mock_gather = self._run_main(
            ["--model-id", "mlx-community/DoesNotExist"],
            count_return=(0, per_worker),
        )
        mock_gather.assert_not_called()

    def test_partial_shards_skips_gather(self):
        workers = _load_workers()
        per_worker = [
            {
                "rank": w["rank"],
                "host": w.get("host") or w.get("device"),
                "ip": w["ip"],
                "found": i % 2,
            }
            for i, w in enumerate(workers)
        ]
        partial_found = sum(e["found"] for e in per_worker)
        mock_count, mock_gather = self._run_main(
            ["--model-id", "mlx-community/PartialModel"],
            count_return=(partial_found, per_worker),
        )
        mock_gather.assert_not_called()

    def test_all_shards_calls_gather(self):
        workers = _load_workers()
        per_worker = [
            {
                "rank": w["rank"],
                "host": w.get("host") or w.get("device"),
                "ip": w["ip"],
                "found": 1,
            }
            for w in workers
        ]
        gather_return = {
            "gathered": [
                {
                    "rank": w["rank"],
                    "host": w.get("host") or w.get("device"),
                    "shard_path": f"/tmp/shard_{w['rank']}.safetensors",
                }
                for w in workers
            ],
            "save_path": "/tmp/out.safetensors",
        }
        mock_count, mock_gather = self._run_main(
            ["--model-id", "mlx-community/SmolLM2-135M-Instruct"],
            count_return=(len(workers), per_worker),
            gather_return=gather_return,
        )
        mock_gather.assert_called_once_with(
            model_id="mlx-community/SmolLM2-135M-Instruct"
        )

    def test_model_id_converted_for_count(self):
        workers = _load_workers()
        per_worker = [
            {
                "rank": w["rank"],
                "host": w.get("host") or w.get("device"),
                "ip": w["ip"],
                "found": 0,
            }
            for w in workers
        ]
        mock_count, _ = self._run_main(
            ["--model-id", "mlx-community/Qwen2.5-0.5B"],
            count_return=(0, per_worker),
        )
        name_arg = mock_count.call_args[0][0]
        assert name_arg == "mlx-community--Qwen2.5-0.5B"


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
        with _CONFIG_PATH.open() as f:
            cfg = yaml.safe_load(f)
        return Path(cfg["data_path"]).parent.name

    def test_known_model_found_on_all_workers(self, workers, known_model):
        total, results = _count_remote_shards(known_model, workers)
        for w in results:
            assert w["found"] >= 1, (
                f"rank {w['rank']} ({w['host']} @ {w['ip']}) reported 0 shards "
                f"for {known_model} — was launch.sh run first?"
            )
        assert total == len(workers)

    def test_unknown_model_returns_zero(self, workers):
        total, results = _count_remote_shards(
            "mlx-community--DoesNotExistModel", workers
        )
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
