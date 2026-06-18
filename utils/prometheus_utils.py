"""All Prometheus metric definitions for smoltorrent.

Grouped by subsystem:
  - TCP networking  (used by networking/send_receive.py)
  - API             (used by backend/api.py)
  - Worker          (used by algorithms/SyncPS/worker.py)
  - Watcher         (used by watcher/watch.py)

Importing prometheus_client is optional — Pi workers don't have it installed.
Every subsystem checks HAS_PROM before touching a metric object.
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, make_asgi_app, start_http_server

    # ── TCP networking ────────────────────────────────────────────────────────
    PROM_BYTES_SENT = Counter("smoltorrent_bytes_sent_total", "Total bytes sent over TCP")
    PROM_BYTES_RECV = Counter("smoltorrent_bytes_recv_total", "Total bytes received over TCP")
    PROM_SEND_SECONDS = Histogram(
        "smoltorrent_send_duration_seconds", "Duration of each TCP send",
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300],
    )
    PROM_RECV_SECONDS = Histogram(
        "smoltorrent_recv_duration_seconds", "Duration of each TCP receive",
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300],
    )
    PROM_SEND_BW_MBPS = Gauge("smoltorrent_send_bandwidth_mbps", "Send bandwidth Mbps (rolling)")
    PROM_RECV_BW_MBPS = Gauge("smoltorrent_recv_bandwidth_mbps", "Recv bandwidth Mbps (rolling)")
    PROM_AVG_SEND_LAT_MS = Gauge("smoltorrent_avg_send_latency_ms", "Avg send latency ms (rolling)")
    PROM_AVG_RECV_LAT_MS = Gauge("smoltorrent_avg_recv_latency_ms", "Avg recv latency ms (rolling)")
    PROM_AVG_BUF_KB = Gauge("smoltorrent_avg_buffer_size_kb", "Average TCP message buffer size KB")
    PROM_MAX_BUF_KB = Gauge("smoltorrent_max_buffer_size_kb", "Max TCP message buffer size KB")

    # ── API ───────────────────────────────────────────────────────────────────
    WALL_BUCKETS = [10, 30, 60, 120, 180, 240, 300, 420, 600]
    api_store_ops = Counter("smoltorrent_store_operations_total", "Completed store operations")
    api_gather_ops = Counter("smoltorrent_gather_operations_total", "Completed gather operations")
    api_xfer_errors = Counter("smoltorrent_transfer_errors_total", "Transfer errors by worker rank", ["rank"])
    api_store_wall = Histogram("smoltorrent_store_wall_seconds", "End-to-end wall-clock time of /store-shard", buckets=WALL_BUCKETS)
    api_gather_wall = Histogram("smoltorrent_gather_wall_seconds", "End-to-end wall-clock time of /gather-shards", buckets=WALL_BUCKETS)
    api_process_start = Gauge("smoltorrent_process_start_time_seconds", "Unix timestamp when this process started")
    api_process_start.set(time.time())

    HAS_PROM = True

except ImportError:
    logger.warning("[prom] prometheus_client not available — metrics will not be exposed")
    HAS_PROM = False
    # Stub every name that other modules import unconditionally.
    # All usages are guarded by `if HAS_PROM:` so these are never called.
    PROM_BYTES_SENT = PROM_BYTES_RECV = None
    PROM_SEND_SECONDS = PROM_RECV_SECONDS = None
    PROM_SEND_BW_MBPS = PROM_RECV_BW_MBPS = None
    PROM_AVG_SEND_LAT_MS = PROM_AVG_RECV_LAT_MS = None
    PROM_AVG_BUF_KB = PROM_MAX_BUF_KB = None
    api_store_ops = api_gather_ops = api_xfer_errors = None
    api_store_wall = api_gather_wall = api_process_start = None
    make_asgi_app = None
    start_http_server = None


# ── TCP helpers ───────────────────────────────────────────────────────────────

def update_prom_gauges(metrics: dict) -> None:
    """Push derived network metrics to Prometheus gauges.

    Args:
        metrics: Dict from :meth:`~utils.network_metrics.NetworkMetrics.get_metrics`;
                 only keys that map to a known Prometheus gauge are written.

    Returns:
        None.
    """
    if not HAS_PROM or not metrics:
        return
    if "send_bandwidth_mbps" in metrics and PROM_SEND_BW_MBPS is not None:
        PROM_SEND_BW_MBPS.set(metrics["send_bandwidth_mbps"])
    if "recv_bandwidth_mbps" in metrics and PROM_RECV_BW_MBPS is not None:
        PROM_RECV_BW_MBPS.set(metrics["recv_bandwidth_mbps"])
    if "avg_send_latency_ms" in metrics and PROM_AVG_SEND_LAT_MS is not None:
        PROM_AVG_SEND_LAT_MS.set(metrics["avg_send_latency_ms"])
    if "avg_recv_latency_ms" in metrics and PROM_AVG_RECV_LAT_MS is not None:
        PROM_AVG_RECV_LAT_MS.set(metrics["avg_recv_latency_ms"])
    if "avg_buffer_size_kb" in metrics and PROM_AVG_BUF_KB is not None:
        PROM_AVG_BUF_KB.set(metrics["avg_buffer_size_kb"])
    if "max_buffer_size_kb" in metrics and PROM_MAX_BUF_KB is not None:
        PROM_MAX_BUF_KB.set(metrics["max_buffer_size_kb"])


# ── Worker metrics ────────────────────────────────────────────────────────────

DURATION_BUCKETS = [1, 5, 10, 30, 60, 120, 300]


@dataclass
class WorkerMetrics:
    bytes_recv: "Counter"
    bytes_sent: "Counter"
    store_ops: "Counter"
    send_ops: "Counter"
    store_errors: "Counter"
    store_duration: "Histogram"
    send_duration: "Histogram"


def init_worker_metrics(rank: int) -> Optional["WorkerMetrics"]:
    """Create and expose per-rank Prometheus worker metrics on port 9200+rank.

    Args:
        rank: Integer worker rank; determines the HTTP metrics port (9200 + rank).

    Returns:
        A populated :class:`WorkerMetrics` dataclass, or ``None`` if
        ``prometheus_client`` is not installed.
    """
    if not HAS_PROM:
        logger.warning("[prom] prometheus_client unavailable — worker metrics disabled")
        return None
    port = 9200 + rank
    m = WorkerMetrics(
        bytes_recv=Counter("worker_bytes_recv_total", "Bytes received (store_shard)", ["rank"]),
        bytes_sent=Counter("worker_bytes_sent_total", "Bytes sent (send_shard)", ["rank"]),
        store_ops=Counter("worker_store_ops_total", "Completed store_shard ops", ["rank"]),
        send_ops=Counter("worker_send_ops_total", "Completed send_shard ops", ["rank"]),
        store_errors=Counter("worker_store_errors_total", "Failed store_shard ops", ["rank"]),
        store_duration=Histogram(
            "worker_store_duration_seconds", "store_shard duration", ["rank"], buckets=DURATION_BUCKETS,
        ),
        send_duration=Histogram(
            "worker_send_duration_seconds", "send_shard duration", ["rank"], buckets=DURATION_BUCKETS,
        ),
    )
    if start_http_server is not None:
        start_http_server(port)
    logger.info("[prom] Worker metrics on port %d", port)
    return m


# ── Watcher metrics ───────────────────────────────────────────────────────────


@dataclass
class WatcherMetrics:
    up: "Gauge"
    syncs_total: "Counter"
    last_sync: "Gauge"


def init_watcher_metrics(port: int = 8001) -> Optional["WatcherMetrics"]:
    """Create and expose watcher Prometheus metrics on *port*.

    Args:
        port: HTTP port on which to serve ``/metrics`` (default 8001).

    Returns:
        A populated :class:`WatcherMetrics` dataclass with ``up`` set to 1,
        or ``None`` if ``prometheus_client`` is not installed.
    """
    if not HAS_PROM:
        logger.warning("[prom] prometheus_client unavailable — watcher metrics disabled")
        return None
    m = WatcherMetrics(
        up=Gauge("watcher_up", "1 while the watcher process is running"),
        syncs_total=Counter("watcher_syncs_total", "Total completed sync cycles"),
        last_sync=Gauge("watcher_last_sync_timestamp", "Unix timestamp of last completed sync cycle"),
    )
    m.up.set(1)
    if start_http_server is not None:
        start_http_server(port)
    logger.info("[prom] Watcher metrics on port %d", port)
    return m
