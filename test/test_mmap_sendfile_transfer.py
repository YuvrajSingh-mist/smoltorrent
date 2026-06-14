"""Tests for the mmap/sendfile shard transfer path.

Unit tests use socket.socketpair() — no cluster, no disk fixtures beyond tmp_path.
Integration tests require a live cluster (pytest -m integration).
Benchmark tests measure throughput of the new path vs the old pickle-based path.

Markers:
  (default)   — unit tests, socketpair only, always fast
  integration — requires workers running (grove start + grove join)
"""

import os
import resource
import socket
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import mmap as _mmap

from networking.send_receive import (
    receive_file,
    serve_file,
)
from utils.common_utils import compute_checksum


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pipe() -> tuple[socket.socket, socket.socket]:
    """Connected socketpair — send on [0], receive on [1]."""
    return socket.socketpair()


def _send_raw(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)
    sock.close()


def _serve_in_thread(sock: socket.socket, file_path: str) -> threading.Thread:
    t = threading.Thread(target=serve_file, args=(sock, file_path), daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Unit: receive_file (raw file mode) via socketpair
# ---------------------------------------------------------------------------


class TestReceiveFileMmap:
    def test_round_trip_small(self, tmp_path):
        data = b"smoltorrent shard bytes" * 50
        send, recv = _pipe()
        threading.Thread(target=_send_raw, args=(send, data), daemon=True).start()

        dest = tmp_path / "out.bin"
        receive_file(recv, str(dest))
        recv.close()

        assert dest.read_bytes() == data

    def test_round_trip_1mb(self, tmp_path):
        data = os.urandom(1 * 1024 * 1024)
        send, recv = _pipe()
        threading.Thread(target=_send_raw, args=(send, data), daemon=True).start()

        dest = tmp_path / "out.bin"
        receive_file(recv, str(dest))
        recv.close()

        assert dest.read_bytes() == data

    def test_round_trip_4mb(self, tmp_path):
        data = os.urandom(4 * 1024 * 1024)
        send, recv = _pipe()
        threading.Thread(target=_send_raw, args=(send, data), daemon=True).start()

        dest = tmp_path / "out.bin"
        receive_file(recv, str(dest))
        recv.close()

        assert dest.stat().st_size == len(data)
        assert dest.read_bytes() == data

    def test_creates_dest_file(self, tmp_path):
        data = b"x" * 256
        send, recv = _pipe()
        threading.Thread(target=_send_raw, args=(send, data), daemon=True).start()

        dest = tmp_path / "new_dir" / "out.bin"
        dest.parent.mkdir(parents=True)
        receive_file(recv, str(dest))
        recv.close()

        assert dest.exists()

    def test_connection_closed_before_header_returns_silently(self, tmp_path):
        send, recv = _pipe()
        send.close()  # close without sending anything

        dest = tmp_path / "out.bin"
        receive_file(recv, str(dest))  # should not raise
        recv.close()

    def test_connection_broken_mid_header_raises(self, tmp_path):
        send, recv = _pipe()

        def send_partial():
            send.sendall(b"\x00\x00")  # only 2 of 4 header bytes
            send.close()

        threading.Thread(target=send_partial, daemon=True).start()

        dest = tmp_path / "out.bin"
        with pytest.raises(ConnectionError):
            receive_file(recv, str(dest))
        recv.close()

    def test_connection_broken_mid_body_raises(self, tmp_path):
        send, recv = _pipe()

        def send_truncated():
            send.sendall(struct.pack(">I", 1024))  # claim 1024 bytes
            send.sendall(b"x" * 256)               # send only 256
            send.close()

        threading.Thread(target=send_truncated, daemon=True).start()

        dest = tmp_path / "out.bin"
        with pytest.raises(ConnectionError):
            receive_file(recv, str(dest))
        recv.close()

    def test_checksum_preserved(self, tmp_path):
        data = os.urandom(512 * 1024)
        src = tmp_path / "src.bin"
        src.write_bytes(data)
        expected = compute_checksum(src)

        send, recv = _pipe()
        threading.Thread(target=_send_raw, args=(send, data), daemon=True).start()

        dest = tmp_path / "dest.bin"
        receive_file(recv, str(dest))
        recv.close()

        assert compute_checksum(dest) == expected


# ---------------------------------------------------------------------------
# Unit: serve_file + receive_file round-trip
# ---------------------------------------------------------------------------


class TestServeSendfileReceiveMmap:
    def test_round_trip_real_file(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(os.urandom(256 * 1024))

        send, recv = _pipe()
        t = _serve_in_thread(send, str(src))

        dest = tmp_path / "dest.bin"
        receive_file(recv, str(dest))
        recv.close()
        t.join(timeout=5)

        assert dest.read_bytes() == src.read_bytes()

    def test_checksum_matches_after_round_trip(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(os.urandom(512 * 1024))
        expected = compute_checksum(src)

        send, recv = _pipe()
        t = _serve_in_thread(send, str(src))

        dest = tmp_path / "dest.bin"
        receive_file(recv, str(dest))
        recv.close()
        t.join(timeout=5)

        assert compute_checksum(dest) == expected

    @pytest.mark.parametrize("size_kb", [1, 64, 512, 4096])
    def test_various_sizes(self, tmp_path, size_kb):
        data = os.urandom(size_kb * 1024)
        src = tmp_path / "src.bin"
        src.write_bytes(data)

        send, recv = _pipe()
        t = _serve_in_thread(send, str(src))

        dest = tmp_path / "dest.bin"
        receive_file(recv, str(dest))
        recv.close()
        t.join(timeout=10)

        assert dest.stat().st_size == len(data)
        assert dest.read_bytes() == data


# ---------------------------------------------------------------------------
# Benchmark: page cache — prove the second read is from RAM not disk
# ---------------------------------------------------------------------------


class TestPageCacheBenchmark:
    """Verify that compute_checksum (pass 1) warms the OS page cache so that the
    subsequent os.sendfile (pass 2) reads the same bytes from RAM.

    We can't drop the page cache without root, so instead we time two back-to-back
    reads of the same range. The second read will always be faster if the OS cached
    the first — which is exactly what happens between the hash pass and sendfile pass
    in send_shard_to_worker.
    """

    def test_two_sequential_reads_match_and_report_speedup(self, tmp_path, capsys):
        """Time two back-to-back reads of the same 64 MB range and report the speedup.

        This mirrors the hash pass + sendfile pass in send_shard_to_worker.
        No pass/fail assertion on the ratio — the speedup is platform-dependent:

          macOS M-series NVMe:  the write itself warms the cache, so both passes
                                hit RAM — difference looks small. Dropping cache
                                requires `sudo purge`.
          Pi SD card:           cold read comes from SD, warm read from page cache.
                                The ratio varies by card and kernel — see performance
                                docs for measured values.

        The correctness assertion (checksums match) is the real test here.
        The timing output is for your information when running on the Pi cluster.
        """
        SIZE = 64 * 1024 * 1024  # 64 MB

        src = tmp_path / "bench.bin"
        src.write_bytes(os.urandom(SIZE))

        t0 = time.perf_counter()
        cksum_pass1 = compute_checksum(src, offset=0, length=SIZE)
        pass1_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        cksum_pass2 = compute_checksum(src, offset=0, length=SIZE)
        pass2_s = time.perf_counter() - t0

        # Correctness: two reads of the same range must produce the same hash
        assert cksum_pass1 == cksum_pass2

        mb = SIZE / (1024 * 1024)
        pass1_mbps = mb / pass1_s
        pass2_mbps = mb / pass2_s
        speedup = pass2_mbps / pass1_mbps

        with capsys.disabled():
            print(f"\n[page-cache] {mb:.0f} MB file")
            print(f"  Pass 1: {pass1_mbps:6.0f} MB/s  ({pass1_s * 1000:.0f} ms)")
            print(f"  Pass 2: {pass2_mbps:6.0f} MB/s  ({pass2_s * 1000:.0f} ms)")
            print(f"  Speedup: {speedup:.1f}x  (meaningful on Pi SD card; ~1x on macOS NVMe both-cached)")
            if speedup < 2.0:
                print("  Note: both passes likely hit the page cache (write warmed it).")
                print("  To see the real disk-vs-cache gap on Linux: sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")

    def test_hash_then_sendfile_checksum_matches(self, tmp_path):
        """End-to-end two-pass pipeline: hash range → sendfile same range.

        This is the exact sequence in send_shard_to_worker. Proves that what the master
        hashes in pass 1 is byte-for-byte what the worker receives in pass 2.
        """
        SIZE = 4 * 1024 * 1024        # 4 MB tensor data
        FILE_OFFSET = 512 * 1024      # start mid-file to exercise the offset seek path

        data = os.urandom(FILE_OFFSET + SIZE)
        src = tmp_path / "ckpt.bin"
        src.write_bytes(data)

        # Pass 1: hash only the shard range — no tensor data loaded into a Python object
        expected_checksum = compute_checksum(str(src), offset=FILE_OFFSET, length=SIZE)

        # Pass 2: sendfile the same range over a socketpair
        send_sock, recv_sock = socket.socketpair()
        dest = tmp_path / "shard.bin"

        def _serve():
            serve_file(send_sock, str(src), FILE_OFFSET, SIZE)
            send_sock.close()

        threading.Thread(target=_serve, daemon=True).start()
        receive_file(recv_sock, str(dest))
        recv_sock.close()

        # The received file must hash to the same value the master computed
        assert compute_checksum(dest) == expected_checksum
        # Belt-and-suspenders: raw bytes match the original range exactly
        assert dest.read_bytes() == data[FILE_OFFSET : FILE_OFFSET + SIZE]

# ---------------------------------------------------------------------------
# Unit: receive_file mmap mode — writes bytes at an offset into a pre-alloc'd mmap
# ---------------------------------------------------------------------------


def _serve_range_in_thread(sock: socket.socket, file_path: str, offset: int, length: int) -> threading.Thread:
    t = threading.Thread(target=serve_file, args=(sock, file_path, offset, length), daemon=True)
    t.start()
    return t


class TestReceiveIntoFdOffset:
    def _make_mm(self, tmp_path: Path, size: int):
        f_path = tmp_path / "merged.bin"
        with open(f_path, "wb") as f:
            f.truncate(size)
        fh = open(f_path, "r+b")
        mm = _mmap.mmap(fh.fileno(), size, access=_mmap.ACCESS_WRITE)
        return f_path, mm, fh

    def test_writes_at_correct_offset(self, tmp_path):
        data = os.urandom(64 * 1024)
        src = tmp_path / "src.bin"
        src.write_bytes(data)

        header_size = 128
        total_size = header_size + len(data)
        f_path, mm, fh = self._make_mm(tmp_path, total_size)
        try:
            send, recv = _pipe()
            t = _serve_range_in_thread(send, str(src), 0, len(data))
            receive_file(recv, mm, write_offset=header_size, expected_length=len(data))
            recv.close()
            t.join(timeout=5)
            mm.flush()
            result = f_path.read_bytes()
        finally:
            mm.close()
            fh.close()

        assert result[header_size:] == data
        assert result[:header_size] == b"\x00" * header_size

    def test_announced_length_mismatch_raises(self, tmp_path):
        data = os.urandom(1024)
        src = tmp_path / "src.bin"
        src.write_bytes(data)

        f_path, mm, fh = self._make_mm(tmp_path, 4096)
        try:
            send, recv = _pipe()
            t = _serve_range_in_thread(send, str(src), 0, len(data))
            with pytest.raises(ValueError, match="expected"):
                receive_file(recv, mm, write_offset=0, expected_length=len(data) + 1)
            recv.close()
            t.join(timeout=5)
        finally:
            mm.close()
            fh.close()

    def test_socket_closed_before_header_raises(self, tmp_path):
        f_path, mm, fh = self._make_mm(tmp_path, 1024)
        try:
            send, recv = _pipe()
            send.close()
            with pytest.raises(ConnectionError):
                receive_file(recv, mm, write_offset=0, expected_length=512)
            recv.close()
        finally:
            mm.close()
            fh.close()

    def test_parallel_writes_to_non_overlapping_offsets(self, tmp_path):
        """Two threads writing to separate halves of the same mmap must not interfere."""
        chunk = 256 * 1024
        src_a = tmp_path / "a.bin"
        src_b = tmp_path / "b.bin"
        data_a = os.urandom(chunk)
        data_b = os.urandom(chunk)
        src_a.write_bytes(data_a)
        src_b.write_bytes(data_b)

        total = chunk * 2
        f_path, mm, fh = self._make_mm(tmp_path, total)
        try:
            send_a, recv_a = _pipe()
            send_b, recv_b = _pipe()
            t_a = _serve_range_in_thread(send_a, str(src_a), 0, chunk)
            t_b = _serve_range_in_thread(send_b, str(src_b), 0, chunk)

            threads = [
                threading.Thread(
                    target=receive_file,
                    args=(recv_a, mm),
                    kwargs={"write_offset": 0,     "expected_length": chunk},
                    daemon=True,
                ),
                threading.Thread(
                    target=receive_file,
                    args=(recv_b, mm),
                    kwargs={"write_offset": chunk, "expected_length": chunk},
                    daemon=True,
                ),
            ]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=10)
            recv_a.close(); recv_b.close()
            t_a.join(timeout=5); t_b.join(timeout=5)
            mm.flush()
            result = f_path.read_bytes()
        finally:
            mm.close()
            fh.close()

        assert result[:chunk] == data_a
        assert result[chunk:] == data_b

    def test_checksum_survives_offset_write(self, tmp_path):
        data = os.urandom(512 * 1024)
        src = tmp_path / "src.bin"
        src.write_bytes(data)
        expected = compute_checksum(str(src))

        header_size = 64
        total = header_size + len(data)
        f_path, mm, fh = self._make_mm(tmp_path, total)
        try:
            send, recv = _pipe()
            t = _serve_range_in_thread(send, str(src), 0, len(data))
            receive_file(recv, mm, write_offset=header_size, expected_length=len(data))
            recv.close()
            t.join(timeout=5)
            mm.flush()
        finally:
            mm.close()
            fh.close()

        actual = compute_checksum(str(f_path), offset=header_size, length=len(data))
        assert actual == expected


# ---------------------------------------------------------------------------
# Unit: streaming merge — simulate full gather with N socketpairs
# ---------------------------------------------------------------------------


class TestStreamingMerge:
    """Simulate the coordinator's gather using socketpairs: each 'worker' thread
    serves a shard's tensor bytes; coordinator writes into pre-allocated mmap."""

    def _build_fake_checkpoint(self, tmp_path: Path, num_shards: int) -> tuple:
        """Build a flat binary file and return per-shard byte ranges."""
        shard_size = 128 * 1024
        total_data = shard_size * num_shards
        header_prefix = 64  # simulate safetensors header section

        src = tmp_path / "fake_ckpt.bin"
        payload = os.urandom(total_data)
        src.write_bytes(b"\x00" * header_prefix + payload)

        ranges = [
            {"file_offset": header_prefix + i * shard_size, "length": shard_size}
            for i in range(num_shards)
        ]
        return src, header_prefix, ranges

    def test_two_shard_merge_bytes_match(self, tmp_path):
        src, header_prefix, ranges = self._build_fake_checkpoint(tmp_path, 2)
        src_bytes = src.read_bytes()

        merged_header_size = 32
        total_data = sum(r["length"] for r in ranges)
        total_size = merged_header_size + total_data

        merged = tmp_path / "merged.bin"
        with open(merged, "wb") as f:
            f.truncate(total_size)

        pairs = [_pipe() for _ in ranges]
        with open(merged, "r+b") as f, _mmap.mmap(f.fileno(), total_size) as mm:
            threads = []
            for r, (send, recv) in zip(ranges, pairs):
                write_offset = merged_header_size + (r["file_offset"] - header_prefix)
                threads.append(threading.Thread(
                    target=serve_file,
                    args=(send, str(src), r["file_offset"], r["length"]),
                    daemon=True,
                ))
                threads.append(threading.Thread(
                    target=receive_file,
                    args=(recv, mm),
                    kwargs={"write_offset": write_offset, "expected_length": r["length"]},
                    daemon=True,
                ))
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=10)
            for send, recv in pairs:
                send.close(); recv.close()
            mm.flush()

        result = merged.read_bytes()
        for r in ranges:
            data_start = r["file_offset"] - header_prefix
            expected = src_bytes[r["file_offset"]: r["file_offset"] + r["length"]]
            actual   = result[merged_header_size + data_start:
                              merged_header_size + data_start + r["length"]]
            assert actual == expected

    @pytest.mark.parametrize("num_shards", [2, 4])
    def test_n_shard_merge_no_data_loss(self, tmp_path, num_shards):
        src, header_prefix, ranges = self._build_fake_checkpoint(tmp_path, num_shards)
        src_bytes = src.read_bytes()

        merged_header_size = 8
        total_data = sum(r["length"] for r in ranges)
        total_size = merged_header_size + total_data

        merged = tmp_path / f"merged_{num_shards}.bin"
        with open(merged, "wb") as f:
            f.truncate(total_size)

        pairs = [_pipe() for _ in ranges]
        with open(merged, "r+b") as f, _mmap.mmap(f.fileno(), total_size) as mm:
            threads = []
            for r, (send, recv) in zip(ranges, pairs):
                write_offset = merged_header_size + (r["file_offset"] - header_prefix)
                threads.append(threading.Thread(
                    target=serve_file,
                    args=(send, str(src), r["file_offset"], r["length"]),
                    daemon=True,
                ))
                threads.append(threading.Thread(
                    target=receive_file,
                    args=(recv, mm),
                    kwargs={"write_offset": write_offset, "expected_length": r["length"]},
                    daemon=True,
                ))
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=15)
            for send, recv in pairs:
                send.close(); recv.close()
            mm.flush()

        result = merged.read_bytes()
        original_tensor_section = src_bytes[header_prefix:]
        assert result[merged_header_size:] == original_tensor_section
