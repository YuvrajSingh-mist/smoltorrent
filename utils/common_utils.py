"""Shared tensor utilities: config loading, shard serialisation, chunking, saving, and merging."""

import hashlib
import io
import json
import logging
import os
import platform
import socket
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union, cast

import httpx
import torch
import yaml
from safetensors.torch import load as st_load
from safetensors.torch import load_file as st_load_file
from safetensors.torch import save as st_save
from safetensors.torch import save_file as st_save_file

from utils.dtypes import MLX_TO_TORCH

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"
API_BASE = "http://localhost:8000"


def compute_checksum(
    src: Union[bytes, str, Path],
    offset: int = 0,
    length: Optional[int] = None,
) -> str:
    """SHA-256 in 1 MB chunks. Accepts in-memory bytes or a file path.

    For file paths, ``offset`` and ``length`` restrict hashing to a byte range —
    used to checksum just one shard's tensor data within the original checkpoint
    without loading it into memory.

    Two-pass zero-copy pattern (used in send_shard_to_worker):
      Pass 1 — compute_checksum(ckpt_path, offset, length):
        f.read() pulls the shard's pages from disk into the OS page cache (kernel RAM).
        SHA-256 runs in userspace on each 1 MB chunk then discards it — peak RAM = 1 MB.

      Pass 2 — serve_file_range → os.sendfile(sock_fd, file_fd, offset, length):
        Single syscall. The kernel finds those same pages still in the page cache
        and copies them directly into the socket send buffer — entirely in kernel space,
        Python userspace never touches the bytes again.

    The page cache is the bridge: pass 1 warms it, pass 2 reads from it for free.
    On SD-card Pi workers this is a ~75x speedup (80 MB/s SD vs 6 GB/s RAM).
    """
    h = hashlib.sha256()
    if isinstance(src, bytes):
        end = offset + length if length is not None else len(src)
        for i in range(offset, end, 1 << 20):
            h.update(src[i : min(i + (1 << 20), end)])
    else:
        with open(src, "rb") as f:
            f.seek(offset)
            remaining = length
            while True:
                to_read = min(1 << 20, remaining) if remaining is not None else 1 << 20
                chunk = f.read(to_read)
                if not chunk:
                    break
                h.update(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break
    return h.hexdigest()


def shard_to_bytes(shard: dict) -> bytes:
    """Serialize a shard dict to safetensors bytes. No temp files, no numpy."""
    if IS_MAC:
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


class NamedBytesIO(io.BytesIO):
    """BytesIO subclass with a hard-coded ``.name`` attribute.

    MLX's ``mx.load()`` requires a file-like object with a ``.name`` ending in
    ``.safetensors``; standard ``io.BytesIO`` has no such attribute.
    """

    name = "shard.safetensors"


def shard_from_bytes(data: bytes) -> dict:
    """Deserialize safetensors bytes. Returns MLX arrays on Mac, torch tensors on Pi."""
    if IS_MAC:
        import mlx.core as mx

        return cast(dict, mx.load(NamedBytesIO(data)))
    return cast(dict, st_load(data))



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
        "[model] Downloading tokenizer and config for %s from HuggingFace Hub...", model_id
    )
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest_dir),
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf", "*.ot"],
    )
    logger.info("[model]   metadata written to %s", dest_dir)


IS_MAC = platform.system() == "Darwin"


def save_shard(shard: dict, path: str) -> None:
    """Save shard to disk. Mac uses MLX, Pi uses safetensors.torch (shard is already torch tensors)."""
    if IS_MAC:
        import mlx.core as mx

        mx.save_safetensors(path, shard)
    else:
        st_save_file(shard, path)


def load_tensors(path: Union[str, Path]) -> dict:
    """Load a safetensors file using MLX on macOS, torch on Linux (Pi workers)."""
    if IS_MAC:
        import mlx.core as mx

        return cast(dict, mx.load(str(path)))
    return st_load_file(str(path))


def connect_with_retry(
    ip: str, port: int, rank: int, retries: int = 3, delay: float = 2.0,
    connect_timeout: float = 5.0,
) -> socket.socket:
    """Open a TCP connection to a worker, retrying on failure with exponential backoff."""
    for attempt in range(1, retries + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(connect_timeout)
        try:
            logger.info("[tcp] Connecting to rank %d at %s:%d (attempt %d/%d)", rank, ip, port, attempt, retries)
            sock.connect((ip, port))
            sock.settimeout(None)
            logger.info("[tcp] Connected to rank %d at %s:%d", rank, ip, port)
            return sock
        except (OSError, ConnectionRefusedError) as e:
            sock.close()
            logger.warning("[tcp] Attempt %d/%d failed for rank %d at %s:%d: %s", attempt, retries, rank, ip, port, e)
            if attempt < retries:
                time.sleep(delay * (2 ** (attempt - 1)))
    raise ConnectionError(f"Could not connect to rank {rank} at {ip}:{port} after {retries} attempts")


def model_id_to_dir_name(model_id: str) -> str:
    """Convert a HuggingFace model ID to a safe directory name.

    ``mlx-community/Qwen2.5-0.5B-Instruct-bf16``
    →  ``mlx-community--Qwen2.5-0.5B-Instruct-bf16``
    """
    return model_id.replace("/", "--")


def gather_shards(model_id: str) -> dict:
    """Call POST /gather-shards, collect streamed output, return structured result."""
    cfg = load_config()
    dir_name = model_id_to_dir_name(model_id)
    ckpt_path = str(Path(cfg["ckpt_root"]).expanduser() / dir_name)
    gathered = []
    save_path = ""
    with httpx.stream("POST", f"{API_BASE}/gather-shards",
                      params={"ckpt_path": ckpt_path}, timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("ERROR:"):
                raise httpx.HTTPStatusError(line, request=resp.request, response=resp)
            if "✓ shard" in line:
                gathered.append(line.strip())
            if line.startswith("Done: saved →"):
                save_path = line.split("→", 1)[-1].strip()
    return {"save_path": save_path, "gathered": gathered}


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
    output_dir: Optional[Union[str, Path]] = None,
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

        save_shard(shard, str(shard_path))

        metadata_payload = dict(merged_metadata)
        metadata_payload["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_payload["source_save_path"] = str(base_path)
        metadata_payload["saved_shard_path"] = str(shard_path)
        metadata_payload["config_path"] = str(resolved_config_path)

        metadata_path = shard_path.with_suffix(shard_path.suffix + ".metadata.json")
        metadata_path.write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

        logger.info("[model] Saved shard to %s with metadata %s", shard_path, metadata_path)
        return str(shard_path), str(metadata_path), True, ""

    except Exception as e:
        logger.error("[model] Failed to save shard: %s", e)
        return "", "", False, str(e)


def merge_shards(shards: list[dict]) -> dict:
    """Merge a list of weight-shard dicts into one."""
    merged = {}
    for shard in shards:
        merged.update(shard)
    return merged


def save_merged_model(merged_weights: dict, save_path: Union[str, Path]) -> Path:
    """Save merged weights as a single safetensors file."""
    dest = Path(save_path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_shard(merged_weights, str(dest))
    logger.info("[model] Saved merged model → %s", dest)
    return dest


def main() -> None:
    """Quick smoke-test for chunk_data."""
    data = {f"tensor_{i}": float(i) for i in range(10)}
    n_chunks = 3
    chunks = chunk_data(data, n_chunks)
    print(chunks)

    for i in range(0, 10, 4):
        print(i)



def handle_json_header(ckpt_path: str) -> tuple[dict, int]:
    """Parse the safetensors JSON header without loading tensor data.

    Returns:
        (header_dict, data_section_offset) — data_section_offset is the absolute
        byte position in the file where tensor data begins:
        8 bytes (uint64 header length field) + header_len bytes (JSON).
    """
    with open(ckpt_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def get_shard_ranges(
    header: dict, data_section_offset: int, num_workers: int
) -> tuple[list[dict], list[dict]]:
    """Compute per-shard byte ranges and rebased tensor metadata from a safetensors header.

    Each shard covers a contiguous slice of the tensor data section.
    chunk_data splits tensor keys sequentially, and safetensors stores tensors
    in header order — so each shard's tensors are always contiguous in the file.

    Args:
        header: Full safetensors header dict (from handle_json_header).
        data_section_offset: Absolute file offset where tensor data starts.
        num_workers: Number of shards to produce.

    Returns:
        shard_ranges: list of {"file_offset": int, "length": int} — one per shard,
                      giving the absolute position and byte count in the original file.
        shard_tensor_meta: list of {tensor_name: {dtype, shape, data_offsets}} — one
                           per shard, with offsets rebased to 0 (relative to the start
                           of that shard's tensor data, ready for a new safetensors file).
    """
    # Skip __metadata__ — only distribute actual weight tensors
    weight_keys = {k: v for k, v in header.items() if k != "__metadata__"}
    chunks = chunk_data(weight_keys, num_workers)

    shard_ranges: list[dict] = []
    shard_tensor_meta: list[dict] = []

    for shard_idx in range(len(chunks)):
        tensors = chunks[shard_idx]
        # Contiguous range of this shard's tensors within the data section
        data_start = min(m["data_offsets"][0] for m in tensors.values())
        data_end   = max(m["data_offsets"][1] for m in tensors.values())

        shard_ranges.append({
            "file_offset": data_section_offset + data_start,
            "length": data_end - data_start,
        })

        # Rebase offsets to 0 so the worker can write a standalone safetensors file
        shard_tensor_meta.append({
            name: {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [
                    meta["data_offsets"][0] - data_start,
                    meta["data_offsets"][1] - data_start,
                ],
            }
            for name, meta in tensors.items()
        })

    return shard_ranges, shard_tensor_meta


if __name__ == "__main__":
    main()

