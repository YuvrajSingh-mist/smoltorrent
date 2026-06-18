"""Shared tensor utilities: config loading, shard serialisation, chunking, saving, and merging."""

import hashlib
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
from safetensors.torch import load_file as st_load_file
from safetensors.torch import save_file as st_save_file

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
    """
    src_label = "<bytes>" if isinstance(src, bytes) else str(src)
    logger.debug("[checksum] starting SHA-256 src=%s offset=%d length=%s", src_label, offset, length)

    h = hashlib.sha256()
    bytes_hashed = 0

    if isinstance(src, bytes):
        end = offset + length if length is not None else len(src)
        for i in range(offset, end, 1 << 20):
            chunk = src[i : min(i + (1 << 20), end)]
            h.update(chunk)
            bytes_hashed += len(chunk)
            logger.debug("[checksum] hashed chunk offset=%d size=%d", i, len(chunk))
    else:
        with open(src, "rb") as f:
            f.seek(offset)
            remaining = length
            pos = offset
            while True:
                to_read = min(1 << 20, remaining) if remaining is not None else 1 << 20
                chunk = f.read(to_read)
                if not chunk:
                    break
                h.update(chunk)
                bytes_hashed += len(chunk)
                logger.debug("[checksum] hashed chunk offset=%d size=%d", pos, len(chunk))
                pos += len(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break

    digest = h.hexdigest()
    logger.info("[checksum] complete src=%s bytes=%d digest=%s", src_label, bytes_hashed, digest)
    return digest




def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load and return the YAML config file.

    Args:
        config_path: Path to the YAML file. Defaults to ``configs/config.yaml``.

    Returns:
        Parsed config dict.
    """
    logger.debug("[config] loading path=%s", config_path)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    logger.debug("[config] loaded %d top-level keys from %s", len(cfg) if cfg else 0, config_path)
    return cfg


def fetch_model_metadata(model_id: str, config: dict) -> None:
    """Download tokenizer and config from HuggingFace Hub into the received_model dir.

    Skips all weight files — the merged .safetensors is already there after gather.

    Args:
        model_id: HuggingFace repo ID, e.g. ``"mlx-community/Qwen2.5-0.5B-Instruct-bf16"``.
        config:   Loaded YAML config dict; uses ``config["save_path"]`` as the destination.

    Returns:
        None.
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
    """Save a tensor shard to disk using the platform-appropriate backend.

    Uses MLX on macOS (``mx.save_safetensors``) and safetensors.torch on Linux/Pi.

    Args:
        shard: Dict mapping tensor names to tensor objects (MLX arrays or torch tensors).
        path:  Absolute destination file path (``*.safetensors``).

    Returns:
        None.
    """
    logger.info("[shard] saving %d tensors → %s platform=%s", len(shard), path, "darwin" if IS_MAC else "linux")
    if IS_MAC:
        import mlx.core as mx

        mx.save_safetensors(path, shard)
    else:
        st_save_file(shard, path)
    logger.debug("[shard] saved %s", path)


def load_tensors(path: Union[str, Path]) -> dict:
    """Load a safetensors file using MLX on macOS, torch on Linux (Pi workers).

    Args:
        path: Path to the ``.safetensors`` file to load.

    Returns:
        Dict mapping tensor names to tensor objects (MLX arrays on macOS,
        torch tensors on Linux).
    """
    logger.info("[tensors] loading %s platform=%s", path, "darwin" if IS_MAC else "linux")
    if IS_MAC:
        import mlx.core as mx

        result = cast(dict, mx.load(str(path)))
    else:
        result = st_load_file(str(path))
    logger.debug("[tensors] loaded %d tensors from %s", len(result), path)
    return result


def connect_with_retry(
    ip: str, port: int, rank: int, retries: int = 3, delay: float = 2.0,
    connect_timeout: float = 5.0,
) -> socket.socket:
    """Open a TCP connection to a worker, retrying on failure with exponential backoff.

    Args:
        ip:              Target worker IP address.
        port:            Target worker TCP port.
        rank:            Worker rank (used only for log messages).
        retries:         Maximum number of connection attempts (default 3).
        delay:           Base delay in seconds between retries; doubles each attempt
                         (default 2.0 → 2s, 4s, 8s …).
        connect_timeout: Per-attempt socket timeout in seconds (default 5.0).

    Returns:
        A connected, blocking :class:`socket.socket`.

    Raises:
        ConnectionError: If all *retries* attempts fail.
    """
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
    logger.error("[tcp] Exhausted all %d retries connecting to rank %d at %s:%d", retries, rank, ip, port)
    raise ConnectionError(f"Could not connect to rank {rank} at {ip}:{port} after {retries} attempts")


def model_id_to_dir_name(model_id: str) -> str:
    """Convert a HuggingFace model ID to a safe directory name.

    ``mlx-community/Qwen2.5-0.5B-Instruct-bf16``
    →  ``mlx-community--Qwen2.5-0.5B-Instruct-bf16``

    Args:
        model_id: HuggingFace repo ID containing a ``/`` separator.

    Returns:
        Directory-safe name with ``/`` replaced by ``--``.
    """
    return model_id.replace("/", "--")


def dir_name_to_model_id(dir_name: str) -> str:
    """Reverse of model_id_to_dir_name — restore the HuggingFace repo ID.

    ``mlx-community--Qwen2.5-0.5B-Instruct-bf16``
    →  ``mlx-community/Qwen2.5-0.5B-Instruct-bf16``

    Args:
        dir_name: Filesystem-safe name with ``--`` as the namespace separator.

    Returns:
        HuggingFace repo ID with the first ``--`` replaced back to ``/``.
    """
    return dir_name.replace("--", "/", 1)


def gather_shards(model_id: str) -> dict:
    """Call POST /gather-shards, collect streamed output, and return a structured result.

    Args:
        model_id: HuggingFace repo ID (or local dir name) identifying the checkpoint.

    Returns:
        Dict with keys ``save_path`` (str — path to the merged file) and
        ``gathered`` (list of confirmation strings, one per shard).

    Raises:
        httpx.HTTPStatusError: If the server returns an HTTP error or streams an
            ``ERROR:`` line.
    """
    logger.info("[gather] requesting shard gather for model_id=%s", model_id)
    cfg = load_config()
    dir_name = model_id_to_dir_name(model_id)
    ckpt_path = str(Path(cfg["ckpt_root"]).expanduser() / dir_name)
    logger.debug("[gather] resolved ckpt_path=%s", ckpt_path)
    gathered = []
    save_path = ""
    with httpx.stream("POST", f"{API_BASE}/gather-shards",
                      params={"ckpt_path": ckpt_path}, timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("ERROR:"):
                logger.error("[gather] server error model_id=%s: %s", model_id, line)
                raise httpx.HTTPStatusError(line, request=resp.request, response=resp)
            if "✓ shard" in line:
                logger.debug("[gather] received shard confirmation: %s", line.strip())
                gathered.append(line.strip())
            if line.startswith("Done: saved →"):
                save_path = line.split("→", 1)[-1].strip()
    logger.info("[gather] complete model_id=%s shards=%d save_path=%s", model_id, len(gathered), save_path)
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
    if n_chunks <= 0:
        logger.error("[chunk] n_chunks must be > 0, got %d", n_chunks)
        raise ValueError(f"n_chunks must be > 0, got {n_chunks}")
    input_len = len(data)
    logger.debug("[chunk] splitting %d items into %d chunks type=%s", input_len, n_chunks, "dict" if isinstance(data, dict) else "list")

    if isinstance(data, dict):
        idx = torch.tensor(list(range(len(data))))
        chunked_tensors = torch.chunk(idx, n_chunks)
        for chunk_idx, chunk_tensor in enumerate(chunked_tensors):
            data_chunks[chunk_idx] = {
                k: v
                for item_idx, (k, v) in enumerate(data.items())
                if item_idx in chunk_tensor
            }
            logger.debug("[chunk] chunk %d → %d items", chunk_idx, len(data_chunks[chunk_idx]))
    else:
        idx = torch.tensor(list(range(len(data))))
        chunked_tensors = torch.chunk(idx, n_chunks)
        for chunk_idx, chunk_tensor in enumerate(chunked_tensors):
            data_chunks[chunk_idx] = [data[item_idx] for item_idx in chunk_tensor]
            logger.debug("[chunk] chunk %d → %d items", chunk_idx, len(data_chunks[chunk_idx]))

    logger.debug("[chunk] produced %d chunks from %d items", len(data_chunks), input_len)
    return data_chunks


def save_received_data_shard(
    shard: Any,
    metadata: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    config_path: Optional[str] = None,
) -> tuple[str, str, bool, str]:
    """Save a received shard using the ``save_path`` from config plus optional metadata.

    The shard filename keeps the original base name from config ``save_path`` and appends
    stable metadata key-value pairs before the extension, for example:
    ``model__rank-2__step-11.safetensors``.

    If ``metadata`` is provided, it is merged with useful auto-generated metadata
    (hostname, platform/device info, pid, and UTC save time).
    A sidecar JSON with the same stem is also written for audit/debugging.

    Args:
        shard:       Dict of tensor name → tensor object to save.
        metadata:    Optional mapping of extra metadata fields (e.g. ``{"rank": 2}``).
        output_dir:  Override the destination directory (defaults to config ``save_path``
                     parent).
        config_path: Path to ``config.yaml`` (defaults to project root).

    Returns:
        Tuple of ``(shard_path, metadata_path, success, error_message)``.
        On success, ``shard_path`` and ``metadata_path`` are absolute path strings
        and ``error_message`` is empty.  On failure, both paths are empty strings
        and ``error_message`` contains the exception text.
    """

    logger.info("[shard] save_received_data_shard rank=%s output_dir=%s config_path=%s", metadata.get("rank") if metadata else None, output_dir, config_path)
    try:
        default_config = (
            Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        )
        resolved_config_path = (
            Path(config_path).expanduser() if config_path else default_config
        )
        logger.debug("[shard] resolved config path=%s", resolved_config_path)

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
        logger.debug("[shard] resolved shard_path=%s", shard_path)

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

        logger.info("[shard] saved rank=%s → %s sidecar=%s", rank, shard_path, metadata_path)
        return str(shard_path), str(metadata_path), True, ""

    except Exception as e:
        logger.error("[shard] save_received_data_shard failed rank=%s error=%s", metadata.get("rank") if metadata else None, e, exc_info=True)
        return "", "", False, str(e)




def handle_json_header(ckpt_path: str) -> tuple[dict, int]:
    """Parse the safetensors JSON header without loading tensor data.

    Args:
        ckpt_path: Absolute path to a ``.safetensors`` checkpoint file.

    Returns:
        Tuple of ``(header_dict, data_section_offset)`` — ``data_section_offset``
        is the absolute byte position in the file where tensor data begins:
        8 bytes (uint64 header length field) + header_len bytes (JSON).

    Raises:
        ValueError: If the file is too short, has a zero header length, or the
            header JSON is truncated.
    """
    logger.debug("[header] parsing safetensors header ckpt_path=%s", ckpt_path)
    with open(ckpt_path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            logger.error("[header] not a safetensors file (too short, %d bytes): %s", len(prefix), ckpt_path)
            raise ValueError(f"Not a safetensors file (too short): {ckpt_path}")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len == 0:
            logger.error("[header] zero header_len — sparse or empty file: %s", ckpt_path)
            raise ValueError(f"Not a safetensors file (zero header length — sparse/empty file?): {ckpt_path}")
        raw = f.read(header_len)
        if len(raw) < header_len:
            logger.error("[header] truncated header in %s: expected %d bytes, got %d", ckpt_path, header_len, len(raw))
            raise ValueError(f"Truncated safetensors header in {ckpt_path}")
        header = json.loads(raw)
    data_offset = 8 + header_len
    n_tensors = len({k for k in header if k != "__metadata__"})
    logger.info("[header] parsed ckpt=%s header_len=%d tensors=%d data_offset=%d", ckpt_path, header_len, n_tensors, data_offset)
    return header, data_offset


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
    # Skip __metadata__ — only distribute actual weight tensors.
    # Sort by data_offsets[0] so each chunk is a contiguous byte range in the file.
    # Safetensors stores tensors in offset order, but the header dict may be alphabetical;
    # chunking by name order produces overlapping shard ranges for many models.
    weight_keys = dict(sorted(
        ((k, v) for k, v in header.items() if k != "__metadata__"),
        key=lambda item: item[1]["data_offsets"][0],
    ))
    logger.info("[ranges] computing shard ranges tensors=%d num_workers=%d data_section_offset=%d", len(weight_keys), num_workers, data_section_offset)
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
        logger.debug("[ranges] shard %d tensors=%d file_offset=%d length=%d", shard_idx, len(tensors), data_section_offset + data_start, data_end - data_start)

    logger.info("[ranges] computed %d shard ranges from %d tensors", len(shard_ranges), len(weight_keys))
    return shard_ranges, shard_tensor_meta
