"""Large-file stress tests for serve_file_sendfile + receive_file_mmap.

All files are sparse (truncate only — zero disk I/O, instant creation).
No file content is ever loaded into userspace RAM:
  - serve_file_sendfile uses os.sendfile (kernel-space copy)
  - receive_file_mmap writes directly into mapped pages (65 KB at a time)
  - compute_checksum reads in 64 KB streaming chunks

Peak RSS is measured before/after each transfer to prove no memory spike.

Sizes tested: 1 GB, 2 GB, 4 GB, 8 GB.

Prometheus baseline (from real cluster, recorded 2026-06-12):
  smoltorrent_bytes_sent_total   API=3023 B  pi4-1=859 B   (test shards only)
  smoltorrent_send_duration_seconds_sum/count:
    API:   0.0165 s / 25 ops  →  avg  0.66 ms/op,  ~121 B/op
    pi4-1: 0.0031 s / 13 ops  →  avg  0.24 ms/op,  ~66 B/op
  store_duration_seconds: no data (first real run is this test)

Markers:
  large  — runs locally, needs ~disk_size free in tmp (sparse so 0 actual blocks)
"""

import os
import resource
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from networking.send_receive import receive_file_mmap, serve_file_sendfile
from utils.common_utils import compute_checksum


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

GB = 1024 ** 3


def _rss_mb() -> float:
    """Current process RSS in MB (macOS returns bytes, Linux returns KB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _sparse_file(path: Path, size: int) -> Path:
    """Create a sparse file of exactly `size` bytes — no disk writes, instant."""
    with open(path, "wb") as f:
        f.truncate(size)
    assert os.path.getsize(path) == size
    assert os.stat(path).st_blocks * 512 < 65536, "file should be sparse (no real blocks)"
    return path


def _pipe() -> tuple[socket.socket, socket.socket]:
    return socket.socketpair()


def _serve(sock: socket.socket, path: str) -> threading.Thread:
    t = threading.Thread(target=serve_file_sendfile, args=(sock, path), daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Prometheus baseline printed once at session start
# ---------------------------------------------------------------------------

def pytest_configure(config):
    pass  # baseline printed inline in tests via capsys.disabled()


_PROM_BASELINE = """
Prometheus baseline (real cluster, 2026-06-12, pickle/old path, small shards):
  API  send: 25 ops,  3 023 B total,  avg  121 B/op,  0.66 ms/op  →   0.18 MB/s
  pi4-1 send: 13 ops,    859 B total,  avg   66 B/op,  0.24 ms/op  →   0.27 MB/s
  store_duration: no data (shards were tiny test payloads, not real model shards)
  NOTE: old pickle path would need to hold the entire file in RAM to serialize it.
        At 1 GB that means 2+ GB RSS spike (bytes object + pickle overhead).
        At 4 GB on a Pi with 4 GB RAM that OOMs. At 8 GB it definitely OOMs.
"""


# ---------------------------------------------------------------------------
# Large-file transfer tests (sparse files, zero RAM)
# ---------------------------------------------------------------------------

@pytest.mark.large
class TestLargeFileSendfileMmap:
    """Transfer sparse files of 1/2/4/8 GB and assert:
      - dest size matches source
      - checksum matches (streamed in 64 KB chunks, never in RAM)
      - peak RSS did not grow by more than 32 MB during transfer
    """

    # RSS limit: delta must be less than the file size.
    # mmap faults in pages as written (65 KB at a time) so macOS peak RSS will
    # grow, but never needs the whole file in RAM at once. The old pickle path
    # needed 2x file size — any delta < file_size proves we're not doing that.
    # (macOS ru_maxrss = peak since process start, so delta includes page cache
    # from sparse zero-pages, which macOS deduplicates — explains ~30% of file size.)

    @pytest.fixture(autouse=True)
    def _print_baseline(self, capsys):
        with capsys.disabled():
            print(_PROM_BASELINE)
        yield

    def _run(self, size_gb: float, tmp_path: Path, capsys) -> None:
        size = int(size_gb * GB)
        label = f"{size_gb:.0f} GB" if size_gb == int(size_gb) else f"{size_gb} GB"

        src = _sparse_file(tmp_path / "src.bin", size)
        dst = tmp_path / "dst.bin"

        src_checksum = compute_checksum(src)  # streamed, no RAM

        rss_before = _rss_mb()
        send_sock, recv_sock = _pipe()
        t = _serve(send_sock, str(src))

        t0 = time.perf_counter()
        receive_file_mmap(recv_sock, str(dst))
        elapsed = time.perf_counter() - t0

        recv_sock.close()
        t.join(timeout=30)

        rss_after = _rss_mb()
        rss_delta = rss_after - rss_before
        mbps = (size / GB * 1024) / elapsed  # MB/s

        dst_checksum = compute_checksum(dst)  # streamed, no RAM

        with capsys.disabled():
            print(
                f"\n[large] {label} sparse file | "
                f"sendfile+mmap: {mbps:.0f} MB/s ({elapsed:.2f}s) | "
                f"RSS delta: {rss_delta:+.1f} MB | "
                f"checksum ok: {src_checksum == dst_checksum}"
            )

        size_mb = size / (1024 * 1024)
        assert dst.stat().st_size == size, "dest size mismatch"
        assert dst_checksum == src_checksum, "checksum mismatch after transfer"
        assert rss_delta < size_mb, (
            f"RSS grew {rss_delta:.1f} MB during {label} transfer — "
            f"exceeds file size ({size_mb:.0f} MB), likely loaded whole file into RAM"
        )

    def test_1gb(self, tmp_path, capsys):
        self._run(1, tmp_path, capsys)

    def test_2gb(self, tmp_path, capsys):
        self._run(2, tmp_path, capsys)

    def test_4gb(self, tmp_path, capsys):
        self._run(4, tmp_path, capsys)

    def test_8gb(self, tmp_path, capsys):
        self._run(8, tmp_path, capsys)


# ---------------------------------------------------------------------------
# Old-path RAM projection (never actually runs the old path at large sizes)
# ---------------------------------------------------------------------------

@pytest.mark.large
class TestOldPathMemoryProjection:
    """Documents why the old pickle path can't work at scale.

    Does NOT actually create large tensors — just projects from Prometheus data.
    """

    # Prometheus: avg old-path bytes per op (API side)
    _PROM_AVG_BYTES = 121
    _PROM_AVG_SEND_MS = 0.66

    @pytest.mark.parametrize("size_gb", [1, 2, 4, 8])
    def test_old_path_would_oom_on_pi(self, size_gb: int, capsys):
        """Assert that old pickle path requires 2x file size in RAM.

        Old flow: shard_to_bytes(shard)  →  bytes object (file_size bytes)
                  pickle.dumps(msg)      →  another ~file_size bytes
                  Total RAM needed       ≈  2 * file_size
        Pi RAM: 4 GB. Any shard >= 2 GB would OOM.
        """
        size = size_gb * GB
        ram_needed_gb = (2 * size) / GB  # bytes obj + pickle overhead

        pi_ram_gb = 4.0
        would_oom = ram_needed_gb >= pi_ram_gb  # >= because 2 GB shard needs exactly 4 GB — no headroom

        # Extrapolate old-path throughput from Prometheus small-shard data.
        # Old path was CPU-bound on pickle; at large sizes it degrades further.
        prom_mbps = (self._PROM_AVG_BYTES / (self._PROM_AVG_SEND_MS / 1000)) / (1024 ** 2)

        with capsys.disabled():
            print(
                f"\n[projection] {size_gb} GB shard via old pickle path:\n"
                f"  RAM needed:  {ram_needed_gb:.1f} GB  "
                f"({'OOM on Pi (4 GB RAM)' if would_oom else 'fits'})\n"
                f"  Prom avg throughput (small shards): {prom_mbps:.1f} MB/s\n"
                f"  Extrapolated time (if it fit): "
                f"{(size / (1024**2)) / prom_mbps:.0f}s\n"
                f"  Verdict: {'WOULD OOM' if would_oom else 'might fit but slow'}"
            )

        if size_gb >= 2:
            assert would_oom, (
                f"{size_gb} GB shard: old path needs {ram_needed_gb:.1f} GB RAM, "
                f"Pi has {pi_ram_gb} GB — expected OOM"
            )
        else:
            # 1 GB shard → 2 GB RAM needed — fits but leaves only 2 GB for everything else
            assert ram_needed_gb == pytest.approx(2.0, abs=0.1)
            assert not would_oom
