from pathlib import Path
import shutil

import mlx.core as mx
import pytest
import yaml
from mlx_lm import generate, load

from utils.common_utils import chunk_data, save_received_data_shard

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_node_shards(
    source_weights_path: Path,
    shard_root: Path,
    n_nodes: int,
) -> list[Path]:
    weights = mx.load(str(source_weights_path))
    chunks = chunk_data(weights, n_chunks=n_nodes)

    saved = []
    for rank, shard in chunks.items():
        node_dir = shard_root / f"node-{rank}"
        shard_path, _ = save_received_data_shard(
            shard=shard,
            metadata={"rank": rank, "step": 0, "node": f"node-{rank}"},
            output_dir=str(node_dir),
        )
        saved.append(Path(shard_path))

    return saved


def _collect_latest_node_shards(shard_root: Path) -> list[Path]:
    shard_paths = []
    for node_dir in sorted(shard_root.glob("node-*")):
        node_shards = sorted(node_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
        if not node_shards:
            continue
        shard_paths.append(node_shards[-1])

    if not shard_paths:
        raise RuntimeError(f"No shard files found in {shard_root}")

    return shard_paths


def _load_shard(path: Path) -> dict:
    """Load a safetensors shard, returning mlx.core.array or torch.Tensor values
    depending on what is actually saved in the file.

    mlx.load on a safetensors file returns mlx.core.array values.
    safetensors.torch.load_file returns torch.Tensor values.
    We use mx.load here because the shards were created by mx.save_safetensors.
    """
    return mx.load(str(path))


def _save_merged(weights: dict, path: Path) -> None:
    """Save merged weights, picking the writer by inspecting the actual value type."""
    path.parent.mkdir(parents=True, exist_ok=True)
    first = next(iter(weights.values()), None)
    if isinstance(first, mx.array):
        mx.save_safetensors(str(path), weights)
    else:
        import torch
        if isinstance(first, torch.Tensor):
            from safetensors.torch import save_file
            save_file(weights, str(path))
        else:
            raise TypeError(
                f"Unsupported tensor type in merged weights: {type(first)}. "
                "Expected mlx.core.array or torch.Tensor."
            )


def _gather_shards_to_master(shard_paths: list[Path], merged_weights_path: Path) -> Path:
    merged_weights = {}

    for shard_path in shard_paths:
        shard = _load_shard(shard_path)
        overlapping_keys = set(merged_weights.keys()) & set(shard.keys())
        if overlapping_keys:
            overlap_preview = sorted(list(overlapping_keys))[:5]
            raise ValueError(
                f"Overlapping tensor keys while gathering shards: {overlap_preview}"
            )
        merged_weights.update(shard)

    _save_merged(merged_weights, merged_weights_path)
    return merged_weights_path


def _prepare_master_model_dir(
    source_model_dir: Path,
    merged_weights_path: Path,
    master_model_dir: Path,
) -> Path:
    if master_model_dir.exists():
        shutil.rmtree(master_model_dir)
    master_model_dir.mkdir(parents=True, exist_ok=True)

    # Copy only flat files, skipping subdirs (e.g. .cache) and existing weights
    _SKIP_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}
    for src_file in source_model_dir.iterdir():
        if not src_file.is_file():
            continue
        if src_file.suffix in _SKIP_SUFFIXES or src_file.name.endswith(".index.json"):
            continue
        shutil.copy2(src_file, master_model_dir / src_file.name)

    # Write the merged weights as the single model file
    target_weights = master_model_dir / "model.safetensors"
    shutil.copy2(merged_weights_path, target_weights)

    return master_model_dir


def _format_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    return prompt


@pytest.fixture(scope="module")
def cluster_shards():
    """Create node shards once for the whole module; shared by both integration tests."""
    config = _load_config()

    source_weights_path = ROOT / config["data_path"]
    if not source_weights_path.exists():
        pytest.skip(f"Source model weights not found at {source_weights_path}")

    n_nodes = int(config.get("num_workers", 1))
    if n_nodes <= 0:
        pytest.skip("num_workers must be > 0 to create node shards")

    shard_root = ROOT / "test" / "fixtures" / "received_shards"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    saved_shards = _prepare_node_shards(
        source_weights_path=source_weights_path,
        shard_root=shard_root,
        n_nodes=n_nodes,
    )

    gathered_shard_paths = _collect_latest_node_shards(shard_root)

    yield {
        "source_weights_path": source_weights_path,
        "source_model_dir": source_weights_path.parent,
        "n_nodes": n_nodes,
        "saved_shards": saved_shards,
        "gathered_shard_paths": gathered_shard_paths,
    }


@pytest.mark.integration
def test_node_shards_present_and_valid(cluster_shards) -> None:
    """Each node shard exists, is non-empty, keys are disjoint, and together cover the full model."""
    n_nodes = cluster_shards["n_nodes"]
    gathered_shard_paths = cluster_shards["gathered_shard_paths"]
    source_weights_path = cluster_shards["source_weights_path"]

    assert len(gathered_shard_paths) == n_nodes, (
        f"Expected {n_nodes} shards, found {len(gathered_shard_paths)}"
    )

    all_shard_keys: list[set] = []
    for rank, shard_path in enumerate(gathered_shard_paths, start=1):
        assert shard_path.exists(), f"Shard for rank {rank} not found: {shard_path}"
        assert shard_path.stat().st_size > 0, f"Shard for rank {rank} is empty: {shard_path}"
        shard_weights = _load_shard(shard_path)
        assert shard_weights, f"Shard for rank {rank} loaded with no tensors: {shard_path}"
        all_shard_keys.append(set(shard_weights.keys()))

    seen_keys: set = set()
    for rank, keys in enumerate(all_shard_keys, start=1):
        overlap = seen_keys & keys
        assert not overlap, (
            f"Rank {rank} shard has overlapping keys with a prior shard: {sorted(overlap)[:5]}"
        )
        seen_keys |= keys

    source_weights = _load_shard(source_weights_path)
    assert seen_keys == set(source_weights.keys()), (
        f"Shard keys don't cover the full model. "
        f"Missing: {sorted(set(source_weights.keys()) - seen_keys)[:5]}, "
        f"Extra: {sorted(seen_keys - set(source_weights.keys()))[:5]}"
    )

    print(f"\n--- Shard validation passed ({n_nodes} shards, {len(seen_keys)} total keys) ---")


@pytest.mark.integration
def test_gather_all_node_shards_to_master_and_generate_text(cluster_shards) -> None:
    """Merge all node shards on master and verify the model generates coherent text."""
    source_model_dir = cluster_shards["source_model_dir"]
    gathered_shard_paths = cluster_shards["gathered_shard_paths"]
    n_nodes = cluster_shards["n_nodes"]

    assert len(gathered_shard_paths) == n_nodes

    merged_weights_path = (
        ROOT / "test" / "fixtures" / "master_gathered_artifacts" / "model.safetensors"
    )
    _gather_shards_to_master(gathered_shard_paths, merged_weights_path)

    master_model_dir = ROOT / "test" / "fixtures" / "master_gathered_model"
    _prepare_master_model_dir(source_model_dir, merged_weights_path, master_model_dir)

    model, tokenizer = load(str(master_model_dir))
    prompt = "Explain what a BitTorrent tracker does in one short paragraph."
    formatted_prompt = _format_prompt(tokenizer, prompt)
    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=80,
        verbose=False,
    )

    assert isinstance(response, str)
    assert response.strip()
    print(f"\n--- Generated response ---\n{response.strip()}\n--------------------------")
