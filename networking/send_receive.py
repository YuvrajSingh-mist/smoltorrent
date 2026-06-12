"""Low-level framed TCP messaging with automatic metrics collection.

Messages are length-prefixed (4-byte big-endian header) and pickled.
A global ``NetworkMetrics`` instance tracks bytes and latency for every send/receive.
"""

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



def get_network_metrics(reset: bool = True) -> dict:
    """Return current network metrics — mirrors the smolcluster FSDP API.

    Keys: ``send_bandwidth_mbps``, ``recv_bandwidth_mbps``,
          ``avg_send_latency_ms``, ``avg_recv_latency_ms``,
          ``avg_buffer_size_kb``, ``max_buffer_size_kb``,
          ``total_send_mb``, ``total_recv_mb``.

    Args:
        reset: If True (default), clear accumulators after reading — use for
               periodic interval logging. Pass False for real-time sampling
               without disturbing rolling totals.
    """
    return network_metrics.get_metrics(reset=reset)



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
    # TCP has no message boundaries — receiver can't tell where one message ends.
    # Prepend payload length as 4-byte big-endian uint so receiver knows exactly
    # how many bytes to read. 4 bytes = 32-bit uint = up to ~4 GB per message.
    sock.sendall(struct.pack(">I", len(data)) + data)
    elapsed = (
        time.time() - start_time
    )  # measured after sendall — includes actual wire time
    network_metrics.record_send(len(data), elapsed)
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
    if HAS_PROM:
        PROM_BYTES_RECV.inc(msglen)
        PROM_RECV_SECONDS.observe(elapsed)
        update_prom_gauges(network_metrics.get_metrics(reset=False))
    return result


def serve_file_sendfile(sock: socket.socket, file_path: str) -> int:
    """Send a file over a socket using zero-copy os.sendfile.

    Sends a 4-byte big-endian length header first so the receiver knows
    the file size, then streams the file bytes via sendfile.

    Args:
        sock: Connected blocking socket.
        file_path: Path to the file to send.

    Returns:
        Total bytes sent (excluding the 4-byte header).
    """
    start_time = time.time()
    filesize = os.path.getsize(file_path)
    network_metrics.record_buffer_size(filesize)

    # Send 4-byte length header so receive_file_mmap knows how many bytes to expect
    sock.sendall(struct.pack(">I", filesize))

    offset = 0
    with open(file_path, "rb") as f:
        while True:
            sent = os.sendfile(sock.fileno(), f.fileno(), offset, filesize - offset)
            if sent == 0:
                break
            offset += sent

    elapsed = time.time() - start_time
    network_metrics.record_send(offset, elapsed)
    if HAS_PROM:
        PROM_BYTES_SENT.inc(offset)
        PROM_SEND_SECONDS.observe(elapsed)
        update_prom_gauges(network_metrics.get_metrics(reset=False))

    return offset
    

def receive_file_mmap(sock: socket.socket, dest_path: str) -> None:
    """Receive a file over a socket using memory-mapped I/O.

    Reads the 4-byte length header sent by serve_file_sendfile, then
    recv_into a memoryview of the mmap'd destination file — bytes land
    directly in the file's page cache without an extra userspace copy.

    Args:
        sock: Connected blocking socket.
        dest_path: Path to write the received file.
    """
    start_time = time.time()
    sock.settimeout(None)

    # Read exactly 4 bytes for the length header (recv may return fewer under load)
    hdr = bytearray(4)
    n = sock.recv_into(hdr, 4)
    if not n:
        return
    if n < 4:
        received = n
        while received < 4:
            n = sock.recv_into(memoryview(hdr)[received:], 4 - received)
            if not n:
                raise ConnectionError("Socket closed while reading length header")
            received += n

    filesize = struct.unpack(">I", hdr)[0]
    network_metrics.record_buffer_size(filesize)

    with open(dest_path, "wb") as f:
        f.truncate(filesize)

    with open(dest_path, "r+b") as f:
        with mmap.mmap(f.fileno(), length=filesize, access=mmap.ACCESS_WRITE) as mm:
            # memoryview of the mmap — slicing gives a writable view into the file.
            # mm[offset:] would return a bytes copy; recv_into would write into it
            # and the data would never reach the file.
            view = memoryview(mm)
            try:
                offset = 0
                while offset < filesize:
                    n = sock.recv_into(view[offset:], min(65536, filesize - offset))
                    if not n:
                        raise ConnectionError("Socket connection broken while receiving file")
                    offset += n
                mm.flush()
                
            finally:
                # Python 3.13: mmap.__exit__ calls close() which raises BufferError if
                # any memoryview export is still alive — release unconditionally.
                view.release()
                
    elapsed = time.time() - start_time
    network_metrics.record_recv(filesize, elapsed)
    if HAS_PROM:
        PROM_BYTES_RECV.inc(filesize)
        PROM_RECV_SECONDS.observe(elapsed)
        update_prom_gauges(network_metrics.get_metrics(reset=False))
    