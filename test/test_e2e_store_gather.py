"""End-to-end integration test: pick 2 random live workers, store a model, gather it back.

Does NOT go through the FastAPI server — calls send_shard_to_worker and
gather_shard_data_only directly so it works even when the API is not running.

Run:
    pytest test/test_e2e_store_gather.py -m api -v -s

What it tests:
  - Worker heartbeat + random selection
  - Zero-copy store (sendfile from coordinator → worker mmap)
  - Zero-copy gather (worker sendfile → coordinator mmap, no tensors in RAM)
  - SQLite shard-tracker round-trip (add_shard_header / get_shard_header)
  - Merged file integrity (SHA-256 of tensor section matches original)
"""

from __future__ import annotations

import json
import mmap
import os
import random
import resource
import struct
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from utils.common_utils import compute_checksum, get_shard_ranges, handle_json_header, load_config
from utils.shard_tracker import add_shard_header, get_shard_header, get_replica_map
from utils.worker_ops import gather_shard_data_only, heartbeat_workers, send_shard_to_worker

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
_REL_PATH_TAG = "__e2e_test__/pytest/store_gather"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _alive_workers() -> list[dict]:
    """Return workers from config that respond to heartbeat."""
    cfg = load_config()
    workers = cfg["devices_config"]["workers"]
    dead = heartbeat_workers(workers, timeout=4.0)
    dead_ranks = {d["rank"] for d in dead}
    alive = [w for w in workers if w["rank"] not in dead_ranks]
    return alive


def _find_or_download_model(cfg: dict) -> Path:
    """Return path to a real model.safetensors under ckpt_root.

    Searches ckpt_root first. If nothing is there, downloads
    HuggingFaceTB/SmolLM2-135M-Instruct (≈270 MB) via huggingface_hub.
    """
    ckpt_root = Path(cfg["ckpt_root"]).expanduser()
    candidates = sorted(ckpt_root.rglob("model.safetensors"))
    if candidates:
        print(f"\n[e2e] Using existing model: {candidates[0]}")
        return candidates[0]

    print("\n[e2e] No model found under ckpt_root — downloading SmolLM2-135M-Instruct…")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        pytest.skip("huggingface_hub not installed — pip install huggingface-hub")

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dest = ckpt_root / "HuggingFaceTB--SmolLM2-135M-Instruct" / "base"
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        allow_patterns=["model.safetensors", "config.json"],
    )
    model_file = dest / "model.safetensors"
    if not model_file.exists():
        pytest.skip(f"Download completed but model.safetensors not found at {model_file}")
    print(f"[e2e] Downloaded to {model_file}")
    return model_file


def _prealloc(path: Path, size: int) -> None:
    """Allocate *size* bytes at *path* — posix_fallocate on Linux, truncate on macOS."""
    with open(path, "wb") as f:
        _fa = getattr(os, "posix_fallocate", None)
        if _fa is not None:
            try:
                _fa(f.fileno(), 0, size)
                return
            except OSError:
                pass
        f.truncate(size)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

N_WORKERS = 2  # change this to test with 3 or 4 workers


@pytest.mark.api
def test_store_and_gather_two_random_workers(capsys, tmp_path):
    """
    1. Heartbeat all configured workers — skip if fewer than N_WORKERS are alive.
    2. Pick N_WORKERS at random.
    3. Find / download a model.safetensors.
    4. Parse safetensors header → split into N_WORKERS shard ranges.
    5. Store each shard to its worker (zero-copy sendfile).
    6. Verify shard-tracker has the right entries.
    7. Gather all shards back into a pre-allocated merged file (zero-copy mmap).
    8. Assert merged tensor section is byte-for-byte identical to the original.
    """
    cfg = load_config()

    # --- 1. find alive workers ---
    alive = _alive_workers()
    if len(alive) < N_WORKERS:
        pytest.skip(f"Need at least {N_WORKERS} alive workers, got {len(alive)}: {[w['host'] for w in alive]}")

    # --- 2. pick N_WORKERS at random ---
    chosen = random.sample(alive, N_WORKERS)
    chosen.sort(key=lambda w: w["rank"])
    n = len(chosen)
    print(f"\n[e2e] Alive workers: {[w['host'] for w in alive]}")
    chosen_str = ", ".join(f"rank {w['rank']} ({w['host']})" for w in chosen)
    print(f"[e2e] Chosen for test: {chosen_str}")

    # --- 3. find / download model ---
    model_path = _find_or_download_model(cfg)
    ckpt_root = Path(cfg["ckpt_root"]).expanduser()
    rel_path = str(model_path.parent.relative_to(ckpt_root)) + "/" + _REL_PATH_TAG
    # Use a unique rel_path so repeated test runs don't collide
    rel_path = str(model_path.parent.relative_to(ckpt_root))

    model_size_mb = model_path.stat().st_size / 1024**2
    print(f"[e2e] Model: {model_path} ({model_size_mb:.1f} MB)")

    # --- 4. parse header, compute n-shard ranges ---
    header, data_section_offset = handle_json_header(str(model_path))
    shard_ranges, shard_tensor_meta = get_shard_ranges(header, data_section_offset, num_workers=n)
    total_tensor_bytes = sum(r["length"] for r in shard_ranges)
    shard_sizes = "  ".join(f"Shard {i}: {r['length']/1024**2:.1f} MB" for i, r in enumerate(shard_ranges))
    print(f"[e2e] {shard_sizes}")

    # Precompute original checksum now — warms OS page cache for subsequent sendfile passes.
    original_checksum = compute_checksum(str(model_path), offset=data_section_offset, length=total_tensor_bytes)
    print(f"[e2e] Original tensor checksum: {original_checksum[:16]}…")

    # Store header in tracker so gather can reconstruct the merged layout.
    # Include precomputed shard_ranges and original_checksum so gather can
    # skip recomputing them and run integrity checks.
    add_shard_header(
        shard_key=rel_path,
        header_json=json.dumps(header, separators=(",", ":")),
        data_section_offset=data_section_offset,
        num_workers=n,
        shard_ranges=shard_ranges,
        total_tensor_bytes=total_tensor_bytes,
        original_checksum=original_checksum,
    )

    # --- 5. store ---
    print("\n[e2e] ── STORE ──────────────────────────────────────────────")
    store_results: list[dict] = []
    t_store = time.perf_counter()

    def _store_one(i: int, worker: dict):
        ok, err, result = send_shard_to_worker(
            worker=worker,
            ckpt_path=str(model_path),
            file_offset=shard_ranges[i]["file_offset"],
            length=shard_ranges[i]["length"],
            tensor_meta=shard_tensor_meta[i],
            rel_path=rel_path,
            shard_filename="shard_0.safetensors",
            shard_index=i,
            size_bytes=model_path.stat().st_size,
            source_path=str(model_path),
        )
        return i, worker, ok, err, result

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(_store_one, i, w): (i, w) for i, w in enumerate(chosen)}
        for future in as_completed(futures):
            i, worker, ok, err, result = future.result()
            rank = worker["rank"]
            host = worker.get("host", worker["ip"])
            if ok:
                store_results.append({"rank": rank, "host": host, **result})
                print(f"  ✓ shard {i} → rank {rank} ({host})")
            else:
                pytest.fail(f"Store failed for shard {i} → rank {rank} ({host}): {err}")

    store_elapsed = time.perf_counter() - t_store
    store_mb = total_tensor_bytes / 1024**2
    print(f"  Store: {store_mb:.1f} MB in {store_elapsed:.2f}s "
          f"= {store_mb/store_elapsed:.1f} MB/s")

    assert len(store_results) == n, f"Expected {n} successful stores, got {len(store_results)}"

    # --- 6. verify tracker ---
    stored = get_shard_header(rel_path)
    assert stored is not None, "Shard header not found in tracker after store"
    assert stored["num_workers"] == n
    assert stored["data_section_offset"] == data_section_offset
    print(f"\n[e2e] Tracker: header stored, data_section_offset={stored['data_section_offset']}, num_workers={stored['num_workers']}")

    # --- 7. gather ---
    print("\n[e2e] ── GATHER ─────────────────────────────────────────────")
    save_path = tmp_path / "merged.safetensors"
    header_json_bytes = stored["header_json"].encode()
    merged_header_size = 8 + len(header_json_bytes)
    total_file_size = merged_header_size + total_tensor_bytes

    _prealloc(save_path, total_file_size)
    with open(save_path, "r+b") as f:
        with mmap.mmap(f.fileno(), length=total_file_size, access=mmap.ACCESS_WRITE) as mm:
            # Write the merged safetensors header
            view = memoryview(mm)
            struct.pack_into("<Q", mm, 0, len(header_json_bytes))
            mm[8 : 8 + len(header_json_bytes)] = header_json_bytes
            view.release()

            t_gather = time.perf_counter()
            gather_ok = True

            # Fetch redundancy map once — one DB query for all shard indices.
            replica_map = get_replica_map(rel_path)
            worker_by_rank = {w["rank"]: w for w in chosen}
            print(f"[e2e] Replica map: { {i: [(r['rank'], r['shard_file']) for r in v] for i, v in replica_map.items()} }")

            def _gather_one(i: int):
                write_offset = merged_header_size + (shard_ranges[i]["file_offset"] - data_section_offset)
                data_length = shard_ranges[i]["length"]

                replicas = replica_map.get(i)
                if not replicas:
                    replicas = [{"rank": chosen[i]["rank"], "shard_file": "shard_0.safetensors", "checksum": ""}]

                ok, err = False, "no replicas tried"
                final_rank, final_host, stored_checksum = -1, "unknown", ""
                for rep in replicas:
                    rep_worker = worker_by_rank.get(rep["rank"])
                    if rep_worker is None:
                        continue
                    ok, err = gather_shard_data_only(
                        worker=rep_worker,
                        rel_path=rel_path,
                        merged_mm=mm,
                        write_offset=write_offset,
                        data_length=data_length,
                        shard_filename=rep["shard_file"],
                    )
                    if ok:
                        final_rank = rep_worker["rank"]
                        final_host = rep_worker.get("host", rep_worker.get("ip", ""))
                        stored_checksum = rep.get("checksum", "")
                        if stored_checksum:
                            actual = compute_checksum(str(save_path), offset=write_offset, length=data_length)
                            if actual != stored_checksum:
                                print(f"  ✗ shard {i}: rank {final_rank} checksum mismatch — trying next replica")
                                ok, err = False, f"checksum mismatch from rank {final_rank}"
                                continue
                        break
                    print(f"  ✗ shard {i}: rank {rep_worker['rank']} failed ({err}) — trying next replica")
                return i, final_rank, final_host, ok, err, write_offset, data_length, stored_checksum

            gathered: list[dict] = []
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = {pool.submit(_gather_one, i): i for i in range(n)}
                for future in as_completed(futures):
                    i, rank, host, ok, err, wo, dl, cs = future.result()
                    if ok:
                        gathered.append({"shard_index": i, "rank": rank, "checksum": cs, "write_offset": wo, "data_length": dl})
                        print(f"  ✓ shard {i} ← rank {rank} ({host})")
                    else:
                        print(f"  ✗ shard {i} ← rank {rank} ({host}): {err}")
                        gather_ok = False

            mm.flush()

    gather_elapsed = time.perf_counter() - t_gather
    print(f"  Gather: {store_mb:.1f} MB in {gather_elapsed:.2f}s "
          f"= {store_mb/gather_elapsed:.1f} MB/s")

    assert gather_ok, "One or more shards failed to gather"
    assert save_path.exists()
    assert save_path.stat().st_size == total_file_size

    # --- 8. verify integrity ---
    print("\n[e2e] ── VERIFY ─────────────────────────────────────────────")

    # Per-shard: verify each shard's bytes in the merged file against stored checksum.
    print("  Per-shard integrity:")
    shard_failures: list[str] = []
    for g in sorted(gathered, key=lambda x: x["shard_index"]):
        i = g["shard_index"]
        expected = g["checksum"]
        if not expected:
            print(f"    ~ shard {i}: no stored checksum — skipping")
            continue
        actual = compute_checksum(str(save_path), offset=g["write_offset"], length=g["data_length"])
        if actual == expected:
            print(f"    ✓ shard {i}: {actual[:16]}…")
        else:
            msg = f"shard {i}: expected {expected[:16]}… got {actual[:16]}…"
            print(f"    ✗ {msg}")
            shard_failures.append(msg)

    assert not shard_failures, "Per-shard checksum failures:\n" + "\n".join(shard_failures)

    # Final: tensor section of merged file must match original.
    merged_checksum = compute_checksum(
        str(save_path), offset=merged_header_size, length=total_tensor_bytes
    )
    print(f"  original tensor checksum: {original_checksum[:16]}…")
    print(f"  merged  tensor checksum:  {merged_checksum[:16]}…")
    assert original_checksum == merged_checksum, (
        "Tensor data mismatch: merged file does not match original!\n"
        f"  original: {original_checksum}\n"
        f"  merged:   {merged_checksum}"
    )
    print("  ✓ checksums match — merged file is byte-perfect")

    print(f"\n[e2e] ── SUMMARY ────────────────────────────────────────────")
    print(f"  model:         {model_path.name}  ({model_size_mb:.1f} MB)")
    print(f"  workers used:  {chosen_str}")
    print(f"  store:         {store_mb/store_elapsed:.1f} MB/s  ({store_elapsed:.2f}s)")
    print(f"  gather:        {store_mb/gather_elapsed:.1f} MB/s  ({gather_elapsed:.2f}s)")
    print(f"  integrity:     ✓ SHA-256 match")


# ---------------------------------------------------------------------------
# Helpers for large e2e tests
# ---------------------------------------------------------------------------

_PROM_API = "http://localhost:8000/metrics"
GB = 1024 ** 3


def _prom_metric(name: str) -> float | None:
    try:
        with urllib.request.urlopen(_PROM_API, timeout=2) as r:
            for line in r.read().decode().splitlines():
                if line.startswith(name + " ") or line.startswith(name + "{"):
                    return float(line.split()[-1])
    except Exception:
        return None


class _RssSampler:
    def __init__(self, interval_s: float = 0.25):
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(interval_s,), daemon=True)

    def start(self):
        self._t.start()
        return self

    def stop(self) -> list[float]:
        self._stop.set()
        self._t.join(timeout=2)
        return self.samples

    def _run(self, interval_s: float):
        while not self._stop.is_set():
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
            self.samples.append(mb)
            self._stop.wait(interval_s)


def _make_fake_safetensors(path: Path, data_size: int, label: str, n_tensors: int = 4) -> Path:
    """Create a sparse safetensors file with a valid header but zero tensor bytes.

    The file is allocated with truncate (no actual disk blocks written) so creation
    is instant even at 8 GB. compute_checksum will SHA-256 over zeros, which is
    deterministic and correct for round-trip verification.

    Creates n_tensors equal-sized tensors so chunk_data can split them across
    n_tensors workers without IndexError. Each tensor is <= data_size/n_tensors
    bytes, safely under the uint32 shard-length limit in the wire protocol.
    """
    assert data_size % n_tensors == 0, "data_size must be divisible by n_tensors"
    shard_bytes = data_size // n_tensors  # bytes per tensor
    n_elements = shard_bytes // 4         # float32 = 4 bytes per element

    tensors = {}
    for idx in range(n_tensors):
        offset_start = idx * shard_bytes
        offset_end = offset_start + shard_bytes
        tensors[f"model.layer{idx}.weight"] = {
            "dtype": "F32",
            "shape": [n_elements],
            "data_offsets": [offset_start, offset_end],
        }

    header = {"__metadata__": {"model_type": label, "torch_dtype": "torch.float32"}, **tensors}
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        total = 8 + len(header_bytes) + data_size
        f.truncate(total)
    return path


# ---------------------------------------------------------------------------
# Large e2e: fake safetensors, all 4 workers, RSS + Prometheus monitoring
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestLargeE2EStoreGather:
    """Full store→gather cycle with fake sparse safetensors files.

    Two sizes: 4 GB ("FakeLLM-4B") and 8 GB ("FakeLLM-8B").
    Uses all available workers (up to 4). Monitors master RSS and Prometheus
    throughout both phases and prints a full report at the end.

    Run:
        pytest test/test_e2e_store_gather.py::TestLargeE2EStoreGather -m api -v -s
    """

    PROM_API = _PROM_API

    def _run(self, data_size: int, label: str, tmp_path: Path) -> None:
        cfg = load_config()
        alive = _alive_workers()
        if not alive:
            pytest.skip("No alive workers found")
        workers = sorted(alive, key=lambda w: w["rank"])
        n = len(workers)
        chosen_str = ", ".join(f"rank {w['rank']} ({w['host']})" for w in workers)

        model_path = _make_fake_safetensors(tmp_path / f"{label}.safetensors", data_size, label, n_tensors=n)
        ckpt_root = Path(cfg["ckpt_root"]).expanduser()
        # Store under a synthetic path that won't collide with real checkpoints
        rel_path = f"__test__/pytest/large_e2e/{label}"

        total_mb = data_size / 1024 ** 2
        print(f"\n[large-e2e] ── {label} ({total_mb:.0f} MB) ──────────────────────────")
        print(f"  workers: {chosen_str}")
        print(f"  file:    {model_path} ({model_path.stat().st_size / GB:.2f} GB on disk)")

        # ── parse header ──────────────────────────────────────────────────────
        header, data_section_offset = handle_json_header(str(model_path))
        shard_ranges, shard_tensor_meta = get_shard_ranges(header, data_section_offset, num_workers=n)
        total_tensor_bytes = sum(r["length"] for r in shard_ranges)

        print(f"  shards:  {n}  ×  {total_tensor_bytes / n / 1024**2:.0f} MB each")
        print(f"  computing original checksum (streaming, no RAM)…")
        t_cs = time.perf_counter()
        original_checksum = compute_checksum(str(model_path), offset=data_section_offset, length=total_tensor_bytes)
        print(f"  checksum: {original_checksum[:16]}…  ({time.perf_counter()-t_cs:.1f}s)")

        add_shard_header(
            shard_key=rel_path,
            header_json=json.dumps(header, separators=(",", ":")),
            data_section_offset=data_section_offset,
            num_workers=n,
            shard_ranges=shard_ranges,
            total_tensor_bytes=total_tensor_bytes,
            original_checksum=original_checksum,
        )

        # ── STORE ─────────────────────────────────────────────────────────────
        print(f"\n[large-e2e] ── STORE ────────────────────────────────────────")
        prom_before_store = _prom_metric("smoltorrent_bytes_sent_total")
        rss_store = _RssSampler().start()
        t_store = time.perf_counter()

        store_results: list[dict] = []
        store_errors: list[dict] = []

        def _store_one(i: int, worker: dict):
            ok, err, result = send_shard_to_worker(
                worker=worker,
                ckpt_path=str(model_path),
                file_offset=shard_ranges[i]["file_offset"],
                length=shard_ranges[i]["length"],
                tensor_meta=shard_tensor_meta[i],
                rel_path=rel_path,
                shard_filename="shard_0.safetensors",
                shard_index=i,
                size_bytes=model_path.stat().st_size,
                source_path=str(model_path),
            )
            return i, worker, ok, err, result

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(_store_one, i, w): (i, w) for i, w in enumerate(workers)}
            for future in as_completed(futures):
                i, worker, ok, err, result = future.result()
                rank = worker["rank"]
                host = worker.get("host", worker["ip"])
                if ok:
                    store_results.append({"rank": rank, "host": host, **result})
                    print(f"  ✓ shard {i} → rank {rank} ({host})")
                else:
                    store_errors.append({"rank": rank, "host": host, "error": err})
                    print(f"  ✗ shard {i} → rank {rank} ({host}): {err}")

        store_elapsed = time.perf_counter() - t_store
        store_samples = rss_store.stop()
        prom_after_store = _prom_metric("smoltorrent_bytes_sent_total")

        store_rss_delta = max(store_samples) - min(store_samples) if store_samples else 0
        store_mbps = total_mb / store_elapsed

        print(f"  throughput:  {store_mbps:.1f} MB/s  ({store_elapsed:.1f}s)")
        print(f"  master RSS:  min {min(store_samples, default=0):.0f} MB  "
              f"max {max(store_samples, default=0):.0f} MB  delta {store_rss_delta:+.0f} MB")
        if prom_before_store is not None and prom_after_store is not None:
            print(f"  Prometheus bytes_sent delta: {(prom_after_store - prom_before_store)/1024**2:.1f} MB")

        assert not store_errors, f"Store failures: {store_errors}"
        assert len(store_results) == n

        # ── GATHER ────────────────────────────────────────────────────────────
        print(f"\n[large-e2e] ── GATHER ───────────────────────────────────────")
        stored = get_shard_header(rel_path)
        assert stored is not None

        save_path = tmp_path / "merged.safetensors"
        header_json_bytes = stored["header_json"].encode()
        merged_header_size = 8 + len(header_json_bytes)
        total_file_size = merged_header_size + total_tensor_bytes

        _prealloc(save_path, total_file_size)
        with open(save_path, "r+b") as f:
            with mmap.mmap(f.fileno(), length=total_file_size, access=mmap.ACCESS_WRITE) as mm:
                struct.pack_into("<Q", mm, 0, len(header_json_bytes))
                mm[8: 8 + len(header_json_bytes)] = header_json_bytes

                replica_map = get_replica_map(rel_path)
                worker_by_rank = {w["rank"]: w for w in workers}

                prom_before_gather = _prom_metric("smoltorrent_bytes_sent_total")
                rss_gather = _RssSampler().start()
                t_gather = time.perf_counter()

                def _gather_one(i: int):
                    write_offset = merged_header_size + (shard_ranges[i]["file_offset"] - data_section_offset)
                    data_length = shard_ranges[i]["length"]
                    replicas = replica_map.get(i) or [{"rank": workers[i]["rank"], "shard_file": "shard_0.safetensors", "checksum": ""}]

                    ok, err = False, "no replicas tried"
                    final_rank, final_host, stored_checksum = -1, "unknown", ""
                    for rep in replicas:
                        rep_worker = worker_by_rank.get(rep["rank"])
                        if rep_worker is None:
                            continue
                        ok, err = gather_shard_data_only(
                            worker=rep_worker, rel_path=rel_path, merged_mm=mm,
                            write_offset=write_offset, data_length=data_length,
                            shard_filename=rep["shard_file"],
                        )
                        if ok:
                            final_rank = rep_worker["rank"]
                            final_host = rep_worker.get("host", rep_worker.get("ip", ""))
                            stored_checksum = rep.get("checksum", "")
                            if stored_checksum:
                                actual = compute_checksum(str(save_path), offset=write_offset, length=data_length)
                                if actual != stored_checksum:
                                    print(f"  ✗ shard {i}: rank {final_rank} checksum mismatch — trying next replica")
                                    ok, err = False, f"checksum mismatch from rank {final_rank}"
                                    continue
                            break
                        print(f"  ✗ shard {i}: rank {rep_worker['rank']} failed ({err}) — trying next replica")
                    return i, final_rank, final_host, ok, err, write_offset, data_length, stored_checksum

                gathered: list[dict] = []
                gather_errors: list[dict] = []
                with ThreadPoolExecutor(max_workers=n) as pool:
                    futures = {pool.submit(_gather_one, i): i for i in range(n)}
                    for future in as_completed(futures):
                        i, rank, host, ok, err, wo, dl, cs = future.result()
                        if ok:
                            gathered.append({"shard_index": i, "rank": rank, "checksum": cs, "write_offset": wo, "data_length": dl})
                            print(f"  ✓ shard {i} ← rank {rank} ({host})")
                        else:
                            gather_errors.append({"shard_index": i, "rank": rank, "error": err})
                            print(f"  ✗ shard {i} ← rank {rank} ({host}): {err}")

                mm.flush()

        gather_elapsed = time.perf_counter() - t_gather
        gather_samples = rss_gather.stop()
        prom_after_gather = _prom_metric("smoltorrent_bytes_sent_total")

        gather_rss_delta = max(gather_samples) - min(gather_samples) if gather_samples else 0
        gather_mbps = total_mb / gather_elapsed

        print(f"  throughput:  {gather_mbps:.1f} MB/s  ({gather_elapsed:.1f}s)")
        print(f"  master RSS:  min {min(gather_samples, default=0):.0f} MB  "
              f"max {max(gather_samples, default=0):.0f} MB  delta {gather_rss_delta:+.0f} MB")
        if prom_before_gather is not None and prom_after_gather is not None:
            print(f"  Prometheus bytes_recv delta: {(prom_after_gather - prom_before_gather)/1024**2:.1f} MB")

        assert not gather_errors, f"Gather failures: {gather_errors}"

        # ── VERIFY ────────────────────────────────────────────────────────────
        print(f"\n[large-e2e] ── VERIFY ───────────────────────────────────────")
        print(f"  computing merged checksum…")
        t_cs = time.perf_counter()
        merged_checksum = compute_checksum(str(save_path), offset=merged_header_size, length=total_tensor_bytes)
        print(f"  checksum: {merged_checksum[:16]}…  ({time.perf_counter()-t_cs:.1f}s)")
        assert original_checksum == merged_checksum, (
            f"Tensor data mismatch!\n  original: {original_checksum}\n  merged:   {merged_checksum}"
        )
        print(f"  ✓ SHA-256 match — merged file is byte-perfect")

        # ── SUMMARY ───────────────────────────────────────────────────────────
        print(f"\n[large-e2e] ── SUMMARY ({label}) ─────────────────────────────")
        print(f"  size:          {total_mb:.0f} MB  ({total_mb/1024:.2f} GB)")
        print(f"  workers:       {chosen_str}")
        print(f"  store:         {store_mbps:.1f} MB/s  ({store_elapsed:.1f}s)  RSS delta {store_rss_delta:+.0f} MB")
        print(f"  gather:        {gather_mbps:.1f} MB/s  ({gather_elapsed:.1f}s)  RSS delta {gather_rss_delta:+.0f} MB")
        print(f"  integrity:     ✓ SHA-256 match")

        # Zero-copy assertion: master RSS must never grow by more than the file size
        assert store_rss_delta < total_mb, (
            f"Store RSS grew {store_rss_delta:.0f} MB during {total_mb:.0f} MB transfer — "
            "exceeds file size, likely buffered whole file in RAM"
        )
        assert gather_rss_delta < total_mb, (
            f"Gather RSS grew {gather_rss_delta:.0f} MB during {total_mb:.0f} MB transfer — "
            "exceeds file size, likely buffered whole file in RAM"
        )

    @pytest.mark.api
    def test_4gb(self, tmp_path):
        self._run(data_size=4 * GB, label="FakeLLM-4B", tmp_path=tmp_path)

    @pytest.mark.api
    def test_8gb(self, tmp_path):
        self._run(data_size=8 * GB, label="FakeLLM-8B", tmp_path=tmp_path)
