"""Shared tensor utilities: config loading, shard serialisation, chunking, saving, and merging."""

import hashlib
import io
import json
import logging
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import yaml
from safetensors.torch import load as st_load
from safetensors.torch import load_file as st_load_file
from safetensors.torch import save as st_save
from safetensors.torch import save_file as st_save_file

from utils.dtypes import MLX_TO_TORCH

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"


def compute_checksum(src: bytes | str | Path) -> str:
    """SHA-256 in 64 KB chunks. Accepts in-memory bytes or a file path."""
    h = hashlib.sha256()
    if isinstance(src, bytes):
        for i in range(0, len(src), 65536):
            h.update(src[i : i + 65536])
    else:
        with open(src, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def shard_to_bytes(shard: dict) -> bytes:
    """Serialize a shard dict to safetensors bytes. No temp files, no numpy."""
    if _IS_MAC:
        import mlx.core as mx

        mx.eval(*shard.values())
        torch_shard = {
            k: torch.frombuffer(bytearray(bytes(v)), dtype=MLX_TO_TORCH[v.dtype])
            .reshape(v.shape)
            .clone()
            for k, v in shard.items()
        }
        return st_save(torch_shard)
    return st_save(shard)


class _NamedBytesIO(io.BytesIO):
    """BytesIO subclass with a hard-coded ``.name`` attribute.

    MLX's ``mx.load()`` requires a file-like object with a ``.name`` ending in
    ``.safetensors``; standard ``io.BytesIO`` has no such attribute.
    """

    name = "shard.safetensors"


def shard_from_bytes(data: bytes) -> dict:
    """Deserialize safetensors bytes. Returns MLX arrays on Mac, torch tensors on Pi."""
    if _IS_MAC:
        import mlx.core as mx

        return dict(mx.load(_NamedBytesIO(data)))
    return st_load(data)


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load and return the YAML config file.

    Args:
        config_path: Path to the YAML file. Defaults to ``configs/config.yaml``.

    Returns:
        Parsed config dict.
    """
    with config_path.open() as f:
        return yaml.safe_load(f)


def fetch_model_metadata(model_id: str, config: dict) -> None:
    """Download tokenizer and config from HuggingFace Hub into the received_model dir.

    Skips all weight files — the merged .safetensors is already there after gather.
    """
    from huggingface_hub import snapshot_download

    dest_dir = Path(config["save_path"]).expanduser().parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading tokenizer and config for %s from HuggingFace Hub...", model_id
    )
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest_dir),
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf", "*.ot"],
    )
    logger.info("  metadata written to %s", dest_dir)


_IS_MAC = platform.system() == "Darwin"


def _save_shard(shard: dict, path: str) -> None:
    """Save shard to disk. Mac uses MLX, Pi uses safetensors.torch (shard is already torch tensors)."""
    if _IS_MAC:
        import mlx.core as mx

        mx.save_safetensors(path, shard)
    else:
        st_save_file(shard, path)


def load_tensors(path: str | Path) -> dict:
    """Load a safetensors file using MLX on macOS, torch on Linux (Pi workers)."""
    if _IS_MAC:
        import mlx.core as mx

        return dict(mx.load(str(path)))
    return st_load_file(str(path))


def model_id_to_dir_name(model_id: str) -> str:
    """Convert a HuggingFace model ID to a safe directory name.

    ``mlx-community/Qwen2.5-0.5B-Instruct-bf16``
    →  ``mlx-community--Qwen2.5-0.5B-Instruct-bf16``
    """
    return model_id.replace("/", "--")


def chunk_data(data, n_chunks: int = 10) -> dict:
    """Split a dict or list into ``n_chunks`` roughly equal parts.

    Args:
        data: A dict of tensors or a list. Keys/indices are distributed evenly.
        n_chunks: Number of output chunks. Must be > 0.

    Returns:
        Dict mapping ``chunk_index`` → slice of ``data``.
    """
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
    output_dir: Optional[str | Path] = None,
    config_path: Optional[str] = None,
) -> tuple[str, str, bool, str]:
    """Save a received shard using config ``save_path`` + metadata.

    The shard filename keeps the original base name from config ``save_path`` and appends
    stable metadata key-value pairs before the extension, for example:
    ``model__rank-2__step-11.safetensors``.

    If ``metadata`` is provided, it is merged with useful auto-generated metadata
    (hostname, platform/device info, pid, and UTC save time).
    A sidecar JSON with the same stem is also written for audit/debugging.
    """

    try:
        default_config = (
            Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        )
        resolved_config_path = (
            Path(config_path).expanduser() if config_path else default_config
        )

        if not resolved_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

        with resolved_config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        save_path = config.get("save_path")
        if not save_path:
            raise ValueError(f"'save_path' missing in config: {resolved_config_path}")

        base_path = Path(save_path).expanduser()
        destination_dir = (
            Path(output_dir).expanduser() if output_dir else base_path.parent
        )
        destination_dir.mkdir(parents=True, exist_ok=True)

        extension = "".join(base_path.suffixes)
        base_name = base_path.name[: -len(extension)] if extension else base_path.name

        # Filename: {model_name}_shard_{rank}.safetensors
        rank = metadata.get("rank", "") if metadata else ""
        data_path_str = config.get("data_path", "")
        model_name = Path(data_path_str).parent.name if data_path_str else base_name
        rank_suffix = f"_shard_{rank}" if rank != "" else ""
        shard_filename = f"{model_name}{rank_suffix}{extension}"
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
        return str(shard_path), str(metadata_path), True, ""

    except Exception as e:
        logger.error("Failed to save shard: %s", e)
        return "", "", False, str(e)


def merge_shards(shards: list[dict]) -> dict:
    """Merge a list of weight-shard dicts into one."""
    merged = {}
    for shard in shards:
        merged.update(shard)
    return merged


def save_merged_model(merged_weights: dict, save_path: str | Path) -> Path:
    """Save merged weights as a single safetensors file."""
    dest = Path(save_path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _save_shard(merged_weights, str(dest))
    logger.info("Saved merged model → %s", dest)
    return dest


def main() -> None:
    """Quick smoke-test for chunk_data."""
    data = {f"tensor_{i}": float(i) for i in range(10)}
    n_chunks = 3
    chunks = chunk_data(data, n_chunks)
    print(chunks)

    for i in range(0, 10, 4):
        print(i)


if __name__ == "__main__":
    main()
