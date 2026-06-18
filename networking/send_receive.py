from __future__ import annotations

"""Low-level framed TCP messaging with automatic metrics collection.

Messages are length-prefixed (4-byte big-endian header) and pickled.
A global ``NetworkMetrics`` instance tracks bytes and latency for every send/receive.
"""

import json
import mmap
import pickle
import struct
import socket
import time
from typing import Any, Optional
import logging
import os
from utils.network_metrics import NetworkMetrics
from utils.prometheus_utils import (
    HAS_PROM,
    PROM_BYTES_SENT, PROM_BYTES_RECV,
    PROM_SEND_SECONDS, PROM_RECV_SECONDS,
    update_prom_gauges,
)

logger = logging.getLogger(__name__)

network_metrics = NetworkMetrics()





def send_message(sock: socket.socket, message: Any) -> None:
    """Pickle ``message``, frame it with a 4-byte length header, and send it in full.

    Args:
        sock: Connected blocking socket.
        message: Any pickleable object to send.
    """
    start_time = time.time()
    data = pickle.dumps(message)
    network_metrics.record_buffer_size(len(data))
    sock.settimeout(None)
    logger.debug("[net/msg] sending %d bytes peer=%s", len(data), sock.getpeername())
    # TCP has no message boundaries — receiver can't tell where one message ends.
    # Prepend payload length as 4-byte big-endian uint so receiver knows exactly
    # how many bytes to read. 4 bytes = 32-bit uint = up to ~4 GB per message.
    sock.sendall(struct.pack(">I", len(data)) + data)
    elapsed = time.time() - start_time
    network_metrics.record_send(len(data), elapsed)
    logger.debug("[net/msg] sent %d bytes in %.3fs", len(data), elapsed)
    if HAS_PROM:
        PROM_BYTES_SENT.inc(len(data))
        PROM_SEND_SECONDS.observe(elapsed)
        update_prom_gauges(network_metrics.get_metrics(reset=False))


def receive_message(sock: socket.socket) -> Optional[Any]:
    """Read one framed message from ``sock`` and unpickle it.

    Args:
        sock: Connected blocking socket.

    Returns:
        Unpickled object, or ``None`` if the remote end closed the connection.

    Raises:
        ConnectionError: If the socket closes mid-message.
    """
    start_time = time.time()
    sock.settimeout(None)

    # Read exactly 4 bytes for the length header (recv may return fewer under load)
    hdr = bytearray(4)
    n = sock.recv_into(hdr, 4)
    if not n:
        return None
    if n < 4:
        received = n
        while received < 4:
            n = sock.recv_into(memoryview(hdr)[received:], 4 - received)
            if not n:
                raise ConnectionError("Socket closed while reading length header")
            received += n

    msglen = struct.unpack(">I", hdr)[0]  # unpack → exact byte count to read
    network_metrics.record_buffer_size(msglen)

    # Pre-allocate buffer of exact size; recv_into writes directly into it —
    # zero copies. Old `data += chunk` on immutable bytes caused O(n²) copying
    # (~240 GB of memcpy for a 169 MB shard, turning 2 min transfer into 13 min).
    buf = bytearray(msglen)
    view = memoryview(buf)
    received = 0

    while received < msglen:  # loop until every byte is in — TCP may split delivery
        n = sock.recv_into(view[received:], min(65536, msglen - received))
        if not n:
            raise ConnectionError("Socket connection broken while receiving message")
        received += n

    result = pickle.loads(buf)
    elapsed = time.time() - start_time
    network_metrics.record_recv(msglen, elapsed)
    logger.debug("[net/msg] received %d bytes in %.3fs", msglen, elapsed)
    if HAS_PROM:
        PROM_BYTES_RECV.inc(msglen)
        PROM_RECV_SECONDS.observe(elapsed)
        update_prom_gauges(network_metrics.get_metrics(reset=False))
    return result


# ---------------------------------------------------------------------------
# Private helpers — shared by serve_file and receive_file
# ---------------------------------------------------------------------------

def _recv_length(sock: socket.socket) -> int:
    """Read the 4-byte big-endian length header from *sock*.

    Args:
        sock: Connected blocking socket to read from.

    Returns:
        Unpacked integer byte count, or ``0`` on a clean peer close.

    Raises:
        ConnectionError: If the peer closes mid-header.
    """
    hdr = bytearray(4)
    n = sock.recv_into(hdr, 4)
    if not n:
        return 0
    if n < 4:
        received = n
        while received < 4:
            try:
                n = sock.recv_into(memoryview(hdr)[received:], 4 - received)
                if not n:
                    raise ConnectionError("Socket closed while reading length header")
                received += n
            except Exception as e:
                logger.error("[net/recv_length] error receiving length header: %s", e)
                raise
            
    return struct.unpack(">I", hdr)[0]


def _recv_into(sock: socket.socket, view: memoryview, start: int, length: int) -> int:
    """Receive *length* bytes from *sock* directly into ``view[start:start+length]``.

    Args:
        sock:   Connected blocking socket to read from.
        view:   Writable memoryview backed by a pre-allocated buffer or mmap.
        start:  Byte offset in *view* at which to begin writing.
        length: Exact number of bytes to receive.

    Returns:
        Number of bytes written (always equals *length* on success).

    Raises:
        ConnectionError: If the peer closes the connection before *length* bytes arrive.
    """
    received = 0
    while received < length:
        try:
            n = sock.recv_into(view[start + received:], min(65536, length - received))
            if not n:
                raise ConnectionError("Socket closed during data transfer")
            received += n
        except Exception as e:
            logger.error("[net/recv_into] error receiving data: %s", e)
            raise
        
    return received


# ---------------------------------------------------------------------------
# File transfer — two public functions cover all send/receive cases
# ---------------------------------------------------------------------------

def serve_file(
    sock: socket.socket,
    file_path: str,
    offset: int = 0,
    length: int | None = None,
) -> int:
    """Send a file (or byte range) via zero-copy os.sendfile.

    Sends a 4-byte big-endian length header first so the receiver knows
    how many bytes to expect, then streams the data via sendfile.

    Args:
        sock:      Connected blocking socket.
        file_path: Path to the source file.
        offset:    Byte offset to start from (default 0 = whole file).
        length:    Number of bytes to send (default None = whole file from offset).

    Returns:
        Bytes sent (excluding the 4-byte header).
    """
    start_time = time.time()
    if length is None:
        length = os.path.getsize(file_path) - offset
    network_metrics.record_buffer_size(length)
    logger.info("[net/sendfile] sending %s offset=%d len=%d (%.2f MB) peer=%s",
                file_path, offset, length, length / (1024 * 1024), sock.getpeername())
    sock.sendall(struct.pack(">I", length))
    sent = 0
    try:
        f = open(file_path, "rb")
    except OSError as e:
        logger.error("[net/sendfile] failed to open %s: %s", file_path, e)
        raise
    with f:
        while sent < length:
            try:
                n = os.sendfile(sock.fileno(), f.fileno(), offset + sent, length - sent)
            except OSError as e:
                logger.error(
                    "[net/sendfile] sendfile error at offset=%d remaining=%d: %s",
                    offset + sent, length - sent, e,
                )
                raise
            if n == 0:
                break
            sent += n
    elapsed = time.time() - start_time
    network_metrics.record_send(sent, elapsed)
    if HAS_PROM:
        PROM_BYTES_SENT.inc(sent)
        PROM_SEND_SECONDS.observe(elapsed)
        update_prom_gauges(network_metrics.get_metrics(reset=False))
    mb = sent / (1024 * 1024)
    mbps = (mb * 8) / elapsed if elapsed > 0 else 0
    logger.info("[net/sendfile] sent %.2f MB @ %.2f Mbps (%.1fs)", mb, mbps, elapsed)
    return sent


def receive_file(
    sock: socket.socket,
    dest: "str | mmap.mmap",
    *,
    write_offset: int = 0,
    st_header: dict | None = None,
    expected_length: int | None = None,
) -> tuple[int, int]:
    """Receive bytes from sock into a file or an existing mmap.

    Three modes, selected by the type of *dest* and presence of *st_header*:

    1. ``dest`` is a **path string**, no *st_header* — raw file write:
       reads announced size, mmap-writes bytes to a new file at *dest*.
       (replaces ``receive_file_mmap``)

    2. ``dest`` is a **path string**, *st_header* given — safetensors file write:
       builds ``[uint64 hdr_len][JSON header][tensor bytes]`` — a valid
       safetensors file.  The announced size is the tensor-data length only.
       (replaces ``receive_shard_mmap``)

    3. ``dest`` is an **open mmap** — write into existing mmap at *write_offset*:
       validates announced size against *expected_length* if provided.
       Thread-safe when called with non-overlapping (write_offset, length) ranges.
       (replaces ``receive_into_fd_offset``)

    Args:
        sock:            Connected blocking socket.
        dest:            File path (str) or open read/write mmap.mmap.
        write_offset:    Byte offset in mmap to start writing (mode 3 only).
        st_header:       Rebased tensor metadata dict (mode 2 only).
        expected_length: Validate announced byte count against this (mode 3 only).

    Returns:
        ``(bytes_transferred, header_size)`` — ``header_size`` is non-zero only
        in mode 2 (offset where tensor data starts inside the written file).
    """
    start_time = time.time()
    sock.settimeout(None)

    announced = _recv_length(sock)
    is_mmap = isinstance(dest, mmap.mmap)

    if announced == 0:
        if is_mmap:
            logger.error("[net/recv] socket closed before sending length header; no data written to mmap")
            raise ConnectionError("Socket closed before sending length header")
        return 0, 0

    if is_mmap:
        # Mode 3: write into pre-allocated mmap at write_offset
        logger.debug("[net/recv] mode=mmap write_offset=%d announced=%d", write_offset, announced)
        if expected_length is not None and announced != expected_length:
            raise ValueError(
                f"receive_file: expected {expected_length} B but peer announced {announced} B"
            )
        length = announced
        network_metrics.record_buffer_size(length)
        view = memoryview(dest)
        try:
            received = _recv_into(sock, view, write_offset, length)
        finally:
            view.release()
        header_size = 0

    elif st_header is not None:
        # Mode 2: build a valid safetensors file — header + tensor bytes
        logger.debug("[net/recv] mode=safetensors dest=%s announced=%d", dest, announced)
        tensor_data_len = announced
        header_json = json.dumps(st_header, separators=(",", ":")).encode()
        header_size = 8 + len(header_json)          # uint64 field + JSON
        total_file_size = header_size + tensor_data_len
        network_metrics.record_buffer_size(tensor_data_len)

        # Write header, then pre-allocate space for tensor data
        with open(dest, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))   # 8-byte LE uint64
            f.write(header_json)
            fallocate = getattr(os, "posix_fallocate", None)
            if fallocate is not None:
                try:
                    fallocate(f.fileno(), 0, total_file_size)
                    logger.info("[net/recv] posix_fallocate succeeded for %s size=%d", dest, total_file_size)
                except OSError:
                    logger.error("[net/recv] posix_fallocate failed; falling back to ftruncate")
                    f.truncate(total_file_size)
            else:
                logger.warning("[net/recv] posix_fallocate not available; falling back to ftruncate")
                f.truncate(total_file_size)
                logger.info("[net/recv] ftruncate succeeded for %s size=%d", dest, total_file_size)
                
        # mmap the whole file; recv tensor bytes directly into the data section
        with open(dest, "r+b") as f:
            with mmap.mmap(f.fileno(), length=total_file_size, access=mmap.ACCESS_WRITE) as mm:
                view = memoryview(mm)
                try:
                    received = _recv_into(sock, view, header_size, tensor_data_len)
                    mm.flush()
                finally:
                    view.release()
        length = tensor_data_len

    else:
        # Mode 1: raw file write
        logger.debug("[net/recv] mode=raw dest=%s announced=%d", dest, announced)
        filesize = announced
        network_metrics.record_buffer_size(filesize)
        header_size = 0
        with open(dest, "wb") as f:
            f.truncate(filesize)
        with open(dest, "r+b") as f:
            with mmap.mmap(f.fileno(), length=filesize, access=mmap.ACCESS_WRITE) as mm:
                view = memoryview(mm)
                try:
                    received = _recv_into(sock, view, 0, filesize)
                    mm.flush()
                finally:
                    # Python 3.13: mmap.__exit__ calls close() which raises BufferError if
                    # any memoryview export is still alive — release unconditionally.
                    view.release()
        length = filesize

    elapsed = time.time() - start_time
    network_metrics.record_recv(length, elapsed)
    if HAS_PROM:
        PROM_BYTES_RECV.inc(length)
        PROM_RECV_SECONDS.observe(elapsed)
        update_prom_gauges(network_metrics.get_metrics(reset=False))
    mb = length / (1024 * 1024)
    mbps = (mb * 8) / elapsed if elapsed > 0 else 0
    dest_label = "mmap" if is_mmap else dest
    logger.info("[net/recv] %.2f MB @ %.2f Mbps (%.1fs) dest=%s", mb, mbps, elapsed, dest_label)
    return received, header_size
