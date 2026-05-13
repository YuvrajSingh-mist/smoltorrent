"""Real HTTP tests for the FastAPI server — /gather-shards and /store-shard.

Marker: api — requires `uvicorn backend.api:app` running on localhost:8000.
Run with:  pytest -m api
"""
from pathlib import Path

import httpx
import pytest
import yaml

from main import gather_shards, API_BASE
from utils.common_utils import model_id_to_dir_name

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
    """POST /store-shard: server reads model from config data_path, shards it,
    and pushes each shard to the respective ranked Pi worker via TCP.

    Requires: server running on localhost:8000 AND all Pi workers reachable.
    Run with:  pytest -m api
    """

    @pytest.fixture(scope="class")
    def config(self):
        with _CONFIG_PATH.open() as f:
            return yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def model_id(self, config) -> str:
        return Path(config["data_path"]).parent.name.replace("--", "/", 1)

    def _post(self, model_id: str = None) -> httpx.Response:
        with httpx.Client() as client:
            params = {"model_id": model_id} if model_id else {}
            return client.post(f"{API_BASE}/store-shard", params=params, timeout=120.0)

    def test_success_response_has_required_keys(self, model_id):
        resp = self._post(model_id=model_id)
        resp.raise_for_status()
        assert {"model_name", "num_shards", "sent_to"} <= resp.json().keys()

    def test_sent_to_all_workers(self, model_id, config):
        resp = self._post(model_id=model_id)
        resp.raise_for_status()
        assert len(resp.json()["sent_to"]) == len(config["devices_config"]["workers"])

    def test_model_name_in_response(self, model_id):
        resp = self._post(model_id=model_id)
        resp.raise_for_status()
        assert resp.json()["model_name"] == model_id_to_dir_name(model_id)

    def test_without_model_id_falls_back_to_config(self, config):
        resp = self._post()  # no model_id — server derives from config data_path
        resp.raise_for_status()
        assert resp.json()["model_name"] == Path(config["data_path"]).parent.name
