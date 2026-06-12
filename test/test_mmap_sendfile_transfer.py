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

from networking.send_receive import receive_file_mmap, serve_file_sendfile
from utils.common_utils import compute_checksum, shard_from_bytes, shard_to_bytes


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
    t = threading.Thread(target=serve_file_sendfile, args=(sock, file_path), daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Unit: receive_file_mmap via socketpair
# ---------------------------------------------------------------------------


class TestReceiveFileMmap:
    def test_round_trip_small(self, tmp_path):
        data = b"smoltorrent shard bytes" * 50
        send, recv = _pipe()
        threading.Thread(target=_send_raw, args=(send, data), daemon=True).start()

        dest = tmp_path / "out.bin"
        receive_file_mmap(recv, str(dest))
        recv.close()

        assert dest.read_bytes() == data

    def test_round_trip_1mb(self, tmp_path):
        data = os.urandom(1 * 1024 * 1024)
        send, recv = _pipe()
        threading.Thread(target=_send_raw, args=(send, data), daemon=True).start()

        dest = tmp_path / "out.bin"
        receive_file_mmap(recv, str(dest))
        recv.close()

        assert dest.read_bytes() == data

    def test_round_trip_4mb(self, tmp_path):
        data = os.urandom(4 * 1024 * 1024)
        send, recv = _pipe()
        threading.Thread(target=_send_raw, args=(send, data), daemon=True).start()

        dest = tmp_path / "out.bin"
        receive_file_mmap(recv, str(dest))
        recv.close()

        assert dest.stat().st_size == len(data)
        assert dest.read_bytes() == data

    def test_creates_dest_file(self, tmp_path):
        data = b"x" * 256
        send, recv = _pipe()
        threading.Thread(target=_send_raw, args=(send, data), daemon=True).start()

        dest = tmp_path / "new_dir" / "out.bin"
        dest.parent.mkdir(parents=True)
        receive_file_mmap(recv, str(dest))
        recv.close()

        assert dest.exists()

    def test_connection_closed_before_header_returns_silently(self, tmp_path):
        send, recv = _pipe()
        send.close()  # close without sending anything

        dest = tmp_path / "out.bin"
        receive_file_mmap(recv, str(dest))  # should not raise
        recv.close()

    def test_connection_broken_mid_header_raises(self, tmp_path):
        send, recv = _pipe()

        def send_partial():
            send.sendall(b"\x00\x00")  # only 2 of 4 header bytes
            send.close()

        threading.Thread(target=send_partial, daemon=True).start()

        dest = tmp_path / "out.bin"
        with pytest.raises(ConnectionError):
            receive_file_mmap(recv, str(dest))
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
            receive_file_mmap(recv, str(dest))
        recv.close()

    def test_checksum_preserved(self, tmp_path):
        data = os.urandom(512 * 1024)
        src = tmp_path / "src.bin"
        src.write_bytes(data)
        expected = compute_checksum(src)

        send, recv = _pipe()
        threading.Thread(target=_send_raw, args=(send, data), daemon=True).start()

        dest = tmp_path / "dest.bin"
        receive_file_mmap(recv, str(dest))
        recv.close()

        assert compute_checksum(dest) == expected


# ---------------------------------------------------------------------------
# Unit: serve_file_sendfile + receive_file_mmap round-trip
# ---------------------------------------------------------------------------


class TestServeSendfileReceiveMmap:
    def test_round_trip_real_file(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(os.urandom(256 * 1024))

        send, recv = _pipe()
        t = _serve_in_thread(send, str(src))

        dest = tmp_path / "dest.bin"
        receive_file_mmap(recv, str(dest))
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
        receive_file_mmap(recv, str(dest))
        recv.close()
        t.join(timeout=5)

        assert compute_checksum(dest) == expected

    def test_safetensors_bytes_survive_round_trip(self, tmp_path):
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not available")

        shard = {"layer.weight": mx.ones([32, 32])}
        raw = shard_to_bytes(shard)

        src = tmp_path / "shard.safetensors"
        src.write_bytes(raw)

        send, recv = _pipe()
        t = _serve_in_thread(send, str(src))

        dest = tmp_path / "received.safetensors"
        receive_file_mmap(recv, str(dest))
        recv.close()
        t.join(timeout=5)

        restored = shard_from_bytes(dest.read_bytes())
        assert "layer.weight" in restored

    @pytest.mark.parametrize("size_kb", [1, 64, 512, 4096])
    def test_various_sizes(self, tmp_path, size_kb):
        data = os.urandom(size_kb * 1024)
        src = tmp_path / "src.bin"
        src.write_bytes(data)

        send, recv = _pipe()
        t = _serve_in_thread(send, str(src))

        dest = tmp_path / "dest.bin"
        receive_file_mmap(recv, str(dest))
        recv.close()
        t.join(timeout=10)

        assert dest.stat().st_size == len(data)
        assert dest.read_bytes() == data


# ---------------------------------------------------------------------------
# Benchmark: mmap/sendfile vs old pickle approach (socketpair, no cluster)
# ---------------------------------------------------------------------------


class TestTransferBenchmark:
    """Measures throughput of both paths over a loopback socketpair.

    Not a correctness test — never fails on performance, just prints stats so
    they show up in -s output alongside Prometheus data from real cluster runs.
    """

    def _old_send_recv(self, shard_bytes: bytes, checksum: str, tmp_path: Path) -> float:
        """Simulate the old pickle-over-send_message path. Returns elapsed seconds.

        Measures wire transfer + deserialization cost. Skips the final save_file
        call because that step is disk I/O unrelated to the wire path, and
        safetensors.torch.save_file requires torch tensors (not MLX arrays on Mac).
        """
        from networking.send_receive import send_message, receive_message

        send, recv = _pipe()

        def sender():
            send_message(send, ("store_shard", 1, shard_bytes, checksum, "test/path"))
            send.close()

        t = threading.Thread(target=sender, daemon=True)
        t0 = time.perf_counter()
        t.start()

        msg = receive_message(recv)
        _, rank, received_bytes, received_checksum, rel_path = msg
        # deserialize (old path did this before save_file)
        shard_from_bytes(received_bytes)
        elapsed = time.perf_counter() - t0

        recv.close()
        t.join(timeout=5)
        return elapsed

    def _new_send_recv(self, shard_bytes: bytes, tmp_path: Path) -> float:
        """Simulate the new sendfile+mmap path. Returns elapsed seconds."""
        src = tmp_path / "new_src.safetensors"
        src.write_bytes(shard_bytes)
        dest = tmp_path / "new_dest.safetensors"

        send, recv = _pipe()
        t0 = time.perf_counter()
        t = _serve_in_thread(send, str(src))
        receive_file_mmap(recv, str(dest))
        elapsed = time.perf_counter() - t0

        recv.close()
        t.join(timeout=5)
        return elapsed

    @pytest.mark.parametrize("size_mb", [1, 4, 16])
    def test_throughput_comparison(self, tmp_path, size_mb, capsys):
        try:
            import mlx.core as mx
            from safetensors.torch import save_file
        except ImportError:
            pytest.skip("mlx/safetensors not available")

        dim = int((size_mb * 1024 * 1024 / 4) ** 0.5)
        shard = {"weight": mx.ones([dim, dim])}
        shard_bytes = shard_to_bytes(shard)
        actual_mb = len(shard_bytes) / (1024 * 1024)

        old_path = tmp_path / "old"
        new_path = tmp_path / "new"
        old_path.mkdir()
        new_path.mkdir()

        old_elapsed = self._old_send_recv(shard_bytes, compute_checksum(shard_bytes), old_path)
        new_elapsed = self._new_send_recv(shard_bytes, new_path)

        old_mbps = actual_mb / old_elapsed
        new_mbps = actual_mb / new_elapsed

        with capsys.disabled():
            print(
                f"\n[benchmark] {actual_mb:.1f} MB shard | "
                f"old (pickle+save_file): {old_mbps:.1f} MB/s ({old_elapsed*1000:.0f}ms) | "
                f"new (sendfile+mmap):    {new_mbps:.1f} MB/s ({new_elapsed*1000:.0f}ms) | "
                f"speedup: {new_mbps/old_mbps:.2f}x"
            )


# ---------------------------------------------------------------------------
# Integration: store_shard new protocol against live workers
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"
_FAKE_REL_PATH = "__test__/pytest/mmap_transfer/latest"


def _load_workers() -> list[dict]:
    try:
        import yaml
        with _CONFIG_PATH.open() as f:
            return yaml.safe_load(f)["devices_config"]["workers"]
    except Exception:
        return []


def _connect(worker: dict, timeout: float = 10.0) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((worker["ip"], worker["port"]))
    sock.settimeout(None)
    return sock


def _store_shard(worker: dict, rank: int, shard_bytes: bytes, checksum: str, rel_path: str):
    """New protocol: metadata message then raw bytes."""
    from networking.send_receive import send_message, receive_message

    sock = _connect(worker, timeout=30.0)
    send_message(sock, ("store_shard", rank, checksum, rel_path))
    sock.sendall(struct.pack(">I", len(shard_bytes)))
    sock.sendall(shard_bytes)
    result = receive_message(sock)
    sock.close()
    return result


WORKERS = _load_workers()


@pytest.mark.integration
class TestStoreShardMmap:
    """store_shard end-to-end with the new metadata+mmap protocol."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_workers(self):
        if not WORKERS:
            pytest.skip("No workers in config")

    def test_store_and_ack(self):
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not available")

        worker = WORKERS[0]
        shard = {"test.weight": mx.ones([8, 8])}
        shard_bytes = shard_to_bytes(shard)
        checksum = compute_checksum(shard_bytes)

        result = _store_shard(worker, worker["rank"], shard_bytes, checksum, _FAKE_REL_PATH)

        assert result is not None
        assert result[0] == "store_shard_done", f"Expected store_shard_done, got: {result}"

    def test_shard_path_in_ack(self):
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not available")

        worker = WORKERS[0]
        shard_bytes = shard_to_bytes({"w": mx.zeros([4, 4])})
        checksum = compute_checksum(shard_bytes)

        result = _store_shard(worker, worker["rank"], shard_bytes, checksum, _FAKE_REL_PATH)

        assert result[0] == "store_shard_done"
        _, rank, shard_path = result
        assert "shard.safetensors" in shard_path
        assert str(worker["rank"]) in shard_path

    def test_bad_checksum_rejected(self):
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not available")

        worker = WORKERS[0]
        shard_bytes = shard_to_bytes({"w": mx.zeros([4, 4])})
        bad_checksum = "0" * 64

        result = _store_shard(worker, worker["rank"], shard_bytes, bad_checksum, _FAKE_REL_PATH)

        assert result is not None
        assert result[0] == "store_shard_failed"
        assert "checksum" in result[2].lower()

    def test_all_workers_accept_store(self):
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not available")

        shard_bytes = shard_to_bytes({"w": mx.ones([8, 8])})
        checksum = compute_checksum(shard_bytes)

        for w in WORKERS:
            result = _store_shard(w, w["rank"], shard_bytes, checksum, _FAKE_REL_PATH)
            assert result[0] == "store_shard_done", f"rank {w['rank']} failed: {result}"


@pytest.mark.integration
class TestSendShardSendfile:
    """send_shard end-to-end: store then retrieve using receive_file_mmap."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_workers(self):
        if not WORKERS:
            pytest.skip("No workers in config")

    def test_round_trip_bytes_match(self, tmp_path):
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not available")

        from networking.send_receive import send_message

        worker = WORKERS[0]
        original = {"w": mx.ones([8, 8]) * 42}
        shard_bytes = shard_to_bytes(original)
        checksum = compute_checksum(shard_bytes)

        _store_shard(worker, worker["rank"], shard_bytes, checksum, _FAKE_REL_PATH)

        sock = _connect(worker, timeout=30.0)
        send_message(sock, ("send_shard", worker["rank"], _FAKE_REL_PATH))
        dest = tmp_path / "received.safetensors"
        receive_file_mmap(sock, str(dest))
        sock.close()

        assert dest.exists()
        restored = shard_from_bytes(dest.read_bytes())
        assert "w" in restored

    def test_checksum_survives_round_trip(self, tmp_path):
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not available")

        from networking.send_receive import send_message

        worker = WORKERS[0]
        shard_bytes = shard_to_bytes({"w": mx.ones([16, 16])})
        original_checksum = compute_checksum(shard_bytes)
        _store_shard(worker, worker["rank"], shard_bytes, original_checksum, _FAKE_REL_PATH)

        sock = _connect(worker, timeout=30.0)
        send_message(sock, ("send_shard", worker["rank"], _FAKE_REL_PATH))
        dest = tmp_path / "received.safetensors"
        receive_file_mmap(sock, str(dest))
        sock.close()

        assert compute_checksum(dest) == original_checksum

    def test_throughput_logged(self, tmp_path, capsys):
        """Store a larger shard and report MB/s for the sendfile retrieve path."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("mlx not available")

        from networking.send_receive import send_message

        worker = WORKERS[0]
        shard = {"weight": mx.ones([128, 128])}
        shard_bytes = shard_to_bytes(shard)
        size_mb = len(shard_bytes) / (1024 * 1024)
        checksum = compute_checksum(shard_bytes)

        _store_shard(worker, worker["rank"], shard_bytes, checksum, _FAKE_REL_PATH)

        sock = _connect(worker, timeout=30.0)
        send_message(sock, ("send_shard", worker["rank"], _FAKE_REL_PATH))
        dest = tmp_path / "received.safetensors"
        t0 = time.perf_counter()
        receive_file_mmap(sock, str(dest))
        elapsed = time.perf_counter() - t0
        sock.close()

        mbps = size_mb / elapsed if elapsed > 0 else 0
        with capsys.disabled():
            print(
                f"\n[integration] send_shard sendfile+mmap: "
                f"{size_mb:.2f} MB in {elapsed*1000:.0f}ms = {mbps:.1f} MB/s "
                f"(rank {worker['rank']} @ {worker['ip']})"
            )

        assert dest.exists()
