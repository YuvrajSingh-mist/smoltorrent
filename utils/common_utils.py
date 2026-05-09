import logging
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import mlx.core as mx
import torch
import yaml

logger = logging.getLogger(__name__)


def chunk_data(data, n_chunks: int = 10) -> dict:
    """Split data into chunks using PyTorch's chunk method."""
    
    data_chunks = {}
    
    assert n_chunks > 0, "n_chunks must be greater than 0"
    
    if isinstance(data, dict):
        # Indexing the data for chunking
        idx = torch.tensor(list(range(len(data))))
         
        # Chunk the tensor (automatically handles uneven divisions)
        chunked_tensors = torch.chunk(idx, n_chunks)
        
        # Convert back to dict format

        for chunk_idx, chunk_tensor in enumerate(chunked_tensors):
            
            data_chunks[chunk_idx] = {k : v for item_idx, (k, v) in enumerate(data.items()) if item_idx in chunk_tensor}
            
    else:
        # Indexing the data for chunking
        idx = torch.tensor(list(range(len(data))))
        
        # Chunk the tensor (automatically handles uneven divisions)
        chunked_tensors = torch.chunk(idx, n_chunks)
        
        # Convert back to list format
        for chunk_idx, chunk_tensor in enumerate(chunked_tensors):
            data_chunks[chunk_idx] = [data[chunk_idx] for chunk_idx in chunk_tensor]
        
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

    auto_metadata = {
        "hostname": socket.gethostname(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "device": platform.processor() or platform.machine(),
        "pid": os.getpid(),
    }

    merged_metadata = dict(auto_metadata)
    if metadata:
        merged_metadata.update(dict(metadata))

    # Keep filenames deterministic for a given metadata mapping.
    metadata_items = sorted((str(k), str(v)) for k, v in merged_metadata.items())
    metadata_suffix = "__".join(f"{k}-{v}" for k, v in metadata_items)
    shard_filename = (
        f"{base_name}__{metadata_suffix}{extension}"
        if metadata_suffix
        else base_path.name
    )
    shard_path = destination_dir / shard_filename

    mx.save(str(shard_path), shard)

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