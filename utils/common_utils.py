import logging
import json
import os
import platform
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


try:
    import mlx.core as _mx
except ImportError:
    _mx = None


def _save_shard(shard: dict, path: str) -> None:
    """Save *shard* to *path*, picking the writer based on the actual value type.

    - mlx.core.array → mx.save_safetensors
    - torch.Tensor   → safetensors.torch.save_file
    """
    from safetensors.torch import save_file as _st_save

    first = next(iter(shard.values()), None)
    if _mx is not None and isinstance(first, _mx.array):
        _mx.save_safetensors(path, shard)
    elif isinstance(first, torch.Tensor):
        _st_save(shard, path)
    else:
        raise TypeError(
            f"Unsupported tensor type in shard: {type(first)}. "
            "Expected mlx.core.array or torch.Tensor."
        )


def chunk_data(data, n_chunks: int = 10) -> dict:
    """Split data into chunks using torch."""
    data_chunks = {}
    assert n_chunks > 0, "n_chunks must be greater than 0"

    if isinstance(data, dict):
        idx = torch.tensor(list(range(len(data))))
        chunked_tensors = torch.chunk(idx, n_chunks)
        for chunk_idx, chunk_tensor in enumerate(chunked_tensors):
            data_chunks[chunk_idx] = {
                k: v
                for item_idx, (k, v) in enumerate(data.items())
                if item_idx in chunk_tensor
            }
    else:
        idx = torch.tensor(list(range(len(data))))
        chunked_tensors = torch.chunk(idx, n_chunks)
        for chunk_idx, chunk_tensor in enumerate(chunked_tensors):
            data_chunks[chunk_idx] = [data[item_idx] for item_idx in chunk_tensor]
    return data_chunks


def save_received_data_shard(
    shard: Any,
    metadata: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
) -> tuple[str, str]:
    """Save a received shard using config ``save_path`` + metadata.

    The shard filename keeps the original base name from config ``save_path`` and appends
    stable metadata key-value pairs before the extension, for example:
    ``model__rank-2__step-11.safetensors``.

    If ``metadata`` is provided, it is merged with useful auto-generated metadata
    (hostname, platform/device info, pid, and UTC save time).
    A sidecar JSON with the same stem is also written for audit/debugging.
    """

    default_config = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    resolved_config_path = Path(config_path).expanduser() if config_path else default_config

    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    save_path = config.get("save_path")
    if not save_path:
        raise ValueError(f"'save_path' missing in config: {resolved_config_path}")

    base_path = Path(save_path).expanduser()
    destination_dir = Path(output_dir).expanduser() if output_dir else base_path.parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    extension = "".join(base_path.suffixes)
    base_name = base_path.name[: -len(extension)] if extension else base_path.name

    # Filename: <model>__rank-<n>__<role>.safetensors
    role = metadata.get("role", "shard") if metadata else "shard"
    rank = metadata.get("rank", "") if metadata else ""
    rank_part = f"rank-{rank}__" if rank != "" else ""
    shard_filename = f"{base_name}__{rank_part}{role}{extension}"
    shard_path = destination_dir / shard_filename

    # Full metadata only goes into the sidecar JSON, not the filename.
    auto_metadata = {
        "hostname": socket.gethostname(),
        "platform_machine": platform.machine(),
        "pid": os.getpid(),
    }

    merged_metadata = dict(auto_metadata)
    if metadata:
        merged_metadata.update(dict(metadata))

    _save_shard(shard, str(shard_path))

    metadata_payload = dict(merged_metadata)
    metadata_payload["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_payload["source_save_path"] = str(base_path)
    metadata_payload["saved_shard_path"] = str(shard_path)
    metadata_payload["config_path"] = str(resolved_config_path)

    metadata_path = shard_path.with_suffix(shard_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    logger.info("Saved shard to %s with metadata %s", shard_path, metadata_path)
    return str(shard_path), str(metadata_path)


def merge_shards(shards: list[dict]) -> dict:
    """Merge a list of weight-shard dicts into a single weights dict."""
    merged = {}
    for shard in shards:
        merged.update(shard)
    return merged


def save_full_model(
    merged_weights: dict,
    source_model_dir: str | Path,
    save_path: str | Path,
) -> Path:
    """Save merged weights + companion files (config, tokenizer, etc.) to save_path.

    Writes model.safetensors (pytorch or mlx depending on tensor type) and copies
    every non-weight file from source_model_dir alongside it.
    """
    source = Path(source_model_dir).expanduser()
    dest = Path(save_path).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    # Copy companion files — skip weight files, we're writing those ourselves
    _WEIGHT_SUFFIXES = {".safetensors"}
    for src_file in source.iterdir():
        if src_file.suffix in _WEIGHT_SUFFIXES or src_file.name.endswith(".index.json"):
            continue
        dst_file = dest / src_file.name
        shutil.copy2(src_file, dst_file)
        logger.info("Copied %s → %s", src_file.name, dst_file)

    # Save merged weights
    weights_path = dest / "model.safetensors"
    _save_shard(merged_weights, str(weights_path))
    logger.info("Saved merged model weights → %s", weights_path)

    return dest


def main():
    """Example usage of chunk_data."""
    data = {f"tensor_{i}": float(i) for i in range(10)}
    n_chunks = 3
    chunks = chunk_data(data, n_chunks)
    print(chunks)
    
    for i in range(0, 10, 4):
        print(i)
    
    
if __name__ == "__main__":
    main()