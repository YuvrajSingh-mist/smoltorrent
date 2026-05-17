"""Prometheus metric initialisation for smoltorrent processes.

Each process calls its init function once at startup. The function creates the
metrics, starts the HTTP server, and returns handles the caller uses to record
observations.
"""
import logging
from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

_DURATION_BUCKETS = [1, 5, 10, 30, 60, 120, 300]


@dataclass
class WorkerMetrics:
    bytes_recv:     Counter
    bytes_sent:     Counter
    store_ops:      Counter
    send_ops:       Counter
    store_errors:   Counter
    store_duration: Histogram
    send_duration:  Histogram


@dataclass
class WatcherMetrics:
    up:          Gauge
    syncs_total: Counter
    last_sync:   Gauge


def init_worker_metrics(rank: int) -> WorkerMetrics:
    port = 9200 + rank
    m = WorkerMetrics(
        bytes_recv     = Counter("worker_bytes_recv_total",    "Bytes received (store_shard)", ["rank"]),
        bytes_sent     = Counter("worker_bytes_sent_total",    "Bytes sent (send_shard)",      ["rank"]),
        store_ops      = Counter("worker_store_ops_total",     "Completed store_shard ops",    ["rank"]),
        send_ops       = Counter("worker_send_ops_total",      "Completed send_shard ops",     ["rank"]),
        store_errors   = Counter("worker_store_errors_total",  "Failed store_shard ops",       ["rank"]),
        store_duration = Histogram("worker_store_duration_seconds", "store_shard duration", ["rank"], buckets=_DURATION_BUCKETS),
        send_duration  = Histogram("worker_send_duration_seconds",  "send_shard duration",  ["rank"], buckets=_DURATION_BUCKETS),
    )
    start_http_server(port)
    logger.info("Worker Prometheus metrics on port %d", port)
    return m


def init_watcher_metrics(port: int = 8001) -> WatcherMetrics:
    m = WatcherMetrics(
        up          = Gauge("watcher_up",                   "1 while the watcher process is running"),
        syncs_total = Counter("watcher_syncs_total",        "Total completed sync cycles"),
        last_sync   = Gauge("watcher_last_sync_timestamp",  "Unix timestamp of last completed sync cycle"),
    )
    m.up.set(1)
    start_http_server(port)
    logger.info("Watcher Prometheus metrics on port %d", port)
    return m
