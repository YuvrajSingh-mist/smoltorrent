"""Real HTTP tests for the FastAPI server — /gather-shards and /store-shard.

Marker: api — requires `uvicorn backend.api:app` running on localhost:8000.
Run with:  pytest -m api
"""

from pathlib import Path

import httpx
import pytest
import yaml

from utils.common_utils import API_BASE, gather_shards

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# /gather-shards
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestGatherShards:
    """Requires: server running + shards present on all Pi workers."""

    @pytest.fixture(scope="class")
    def model_id(self) -> str:
        with _CONFIG_PATH.open() as f:
            cfg = yaml.safe_load(f)
        return Path(cfg["data_path"]).parent.name.replace("--", "/", 1)

    def test_success_returns_body(self, model_id):
        result = gather_shards(model_id)
        assert "save_path" in result
        assert "gathered" in result

    def test_gathered_list_nonempty(self, model_id):
        result = gather_shards(model_id)
        assert len(result["gathered"]) > 0, (
            f"No shards gathered for {model_id} — are the workers running?"
        )

    def test_unknown_model_raises_http_error(self):
        with pytest.raises(httpx.HTTPStatusError):
            gather_shards("mlx-community/DoesNotExistModel-xyz")


# ---------------------------------------------------------------------------
# /store-shard
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestStoreShard:
    """POST /store-shard?ckpt_path=<path>: server shards the file and pushes each
    shard to the ranked Pi workers via TCP.

    Requires: server running on localhost:8000 AND all Pi workers reachable.
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

    def _stream_post(self, ckpt_path: str) -> str:
        with httpx.Client(timeout=None) as client:
            with client.stream(
                "POST", f"{API_BASE}/store-shard", params={"ckpt_path": ckpt_path}
            ) as resp:
                resp.raise_for_status()
                return resp.read().decode()

    def test_store_succeeds_no_error(self, ckpt_path):
        body = self._stream_post(ckpt_path)
        assert "ERROR" not in body, f"Store failed:\n{body}"

    def test_store_done_line_present(self, ckpt_path):
        body = self._stream_post(ckpt_path)
        assert "Done:" in body, f"No Done line:\n{body}"

    def test_store_all_workers_acked(self, ckpt_path, config):
        body = self._stream_post(ckpt_path)
        for w in config["devices_config"]["workers"]:
            assert f"rank {w['rank']}" in body, (
                f"rank {w['rank']} not mentioned in store output"
            )
