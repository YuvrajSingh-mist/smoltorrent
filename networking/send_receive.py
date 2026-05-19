"""Low-level framed TCP messaging with automatic metrics collection.

Messages are length-prefixed (4-byte big-endian header) and pickled.
A global ``NetworkMetrics`` instance tracks bytes and latency for every send/receive.
"""

import pickle
import struct
import socket
import time
from typing import Any, Optional
import logging

from utils.network_metrics import NetworkMetrics

logger = logging.getLogger(__name__)

_network_metrics = NetworkMetrics()

# Optional — only available on master (Mac). Pi workers don't expose /metrics.
try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROM_BYTES_SENT = Counter(
        "smoltorrent_bytes_sent_total", "Total bytes sent over TCP"
    )
    _PROM_BYTES_RECV = Counter(
        "smoltorrent_bytes_recv_total", "Total bytes received over TCP"
    )
    _PROM_SEND_SECONDS = Histogram(
        "smoltorrent_send_duration_seconds",
        "Duration of each TCP send",
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300],
    )
    _PROM_RECV_SECONDS = Histogram(
        "smoltorrent_recv_duration_seconds",
        "Duration of each TCP receive",
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300],
    )
    # Derived gauges — same fields as FSDP's get_network_metrics() dict, readable directly in Grafana
    # without needing rate() PromQL on raw counters.
    _PROM_SEND_BW_MBPS = Gauge(
        "smoltorrent_send_bandwidth_mbps", "Send bandwidth Mbps (rolling)"
    )
    _PROM_RECV_BW_MBPS = Gauge(
        "smoltorrent_recv_bandwidth_mbps", "Recv bandwidth Mbps (rolling)"
    )
    _PROM_AVG_SEND_LAT_MS = Gauge(
        "smoltorrent_avg_send_latency_ms", "Avg send latency ms (rolling)"
    )
    _PROM_AVG_RECV_LAT_MS = Gauge(
        "smoltorrent_avg_recv_latency_ms", "Avg recv latency ms (rolling)"
    )
    _PROM_AVG_BUF_KB = Gauge(
        "smoltorrent_avg_buffer_size_kb", "Average TCP message buffer size KB"
    )
    _PROM_MAX_BUF_KB = Gauge(
        "smoltorrent_max_buffer_size_kb", "Max TCP message buffer size KB"
    )
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False


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
    return _network_metrics.get_metrics(reset=reset)


def _update_prom_gauges(metrics: dict) -> None:
    """Push derived metrics to Prometheus gauges."""
    if not _HAS_PROM or not metrics:
        return
    if "send_bandwidth_mbps" in metrics:
        _PROM_SEND_BW_MBPS.set(metrics["send_bandwidth_mbps"])
    if "recv_bandwidth_mbps" in metrics:
        _PROM_RECV_BW_MBPS.set(metrics["recv_bandwidth_mbps"])
    if "avg_send_latency_ms" in metrics:
        _PROM_AVG_SEND_LAT_MS.set(metrics["avg_send_latency_ms"])
    if "avg_recv_latency_ms" in metrics:
        _PROM_AVG_RECV_LAT_MS.set(metrics["avg_recv_latency_ms"])
    if "avg_buffer_size_kb" in metrics:
        _PROM_AVG_BUF_KB.set(metrics["avg_buffer_size_kb"])
    if "max_buffer_size_kb" in metrics:
        _PROM_MAX_BUF_KB.set(metrics["max_buffer_size_kb"])


def send_message(sock: socket.socket, message: Any) -> None:
    """Pickle ``message``, frame it with a 4-byte length header, and send it in full.

    Args:
        sock: Connected blocking socket.
        message: Any pickleable object to send.
    """
    start_time = time.time()
    data = pickle.dumps(message)
    _network_metrics.record_buffer_size(len(data))
    sock.settimeout(None)
    # TCP has no message boundaries — receiver can't tell where one message ends.
    # Prepend payload length as 4-byte big-endian uint so receiver knows exactly
    # how many bytes to read. 4 bytes = 32-bit uint = up to ~4 GB per message.
    sock.sendall(struct.pack(">I", len(data)) + data)
    elapsed = (
        time.time() - start_time
    )  # measured after sendall — includes actual wire time
    _network_metrics.record_send(len(data), elapsed)
    if _HAS_PROM:
        _PROM_BYTES_SENT.inc(len(data))
        _PROM_SEND_SECONDS.observe(elapsed)
        _update_prom_gauges(_network_metrics.get_metrics(reset=False))


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
    _network_metrics.record_buffer_size(msglen)

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
    _network_metrics.record_recv(msglen, elapsed)
    if _HAS_PROM:
        _PROM_BYTES_RECV.inc(msglen)
        _PROM_RECV_SECONDS.observe(elapsed)
        _update_prom_gauges(_network_metrics.get_metrics(reset=False))
    return result
