"""Single entry point for observability setup.

Each process calls one function at startup:

    Master / API server:
        observability.setup_api()                 # starts /metrics endpoint

    Worker (Pi):
        observability.setup_worker(rank, hostname) # starts :92XX metrics + file logging

    Watcher:
        observability.setup_watcher(hostname)      # starts :8001 metrics + file logging

    CLI / one-shot scripts:
        observability.setup_logging()              # coloured console only
"""

import logging
from typing import Optional

from utils.log_utils import setup_logging, setup_cluster_logging
from utils.prometheus_utils import HAS_PROM

logger = logging.getLogger(__name__)


def setup_api() -> None:
    """Init logging for the API process (Prometheus is mounted by FastAPI, not here)."""
    setup_logging()


def setup_worker(rank: int, hostname: str, log_dir: Optional[str] = None) -> object:
    """Init worker metrics server and structured file logging.

    Returns a WorkerMetrics instance (or None if prometheus_client is unavailable).
    """
    setup_logging()
    setup_cluster_logging(
        logger=logging.getLogger("smoltorrent"),
        component="worker",
        rank=rank,
        hostname=hostname,
        log_dir=log_dir,
        algorithm="syncps",
        arch="smoltorrent",
    )
    if not HAS_PROM:
        logger.warning("[obs] prometheus_client not available — worker metrics skipped")
        return None
    from utils.prometheus_utils import init_worker_metrics
    return init_worker_metrics(rank)


def setup_watcher(hostname: Optional[str] = None, log_dir: Optional[str] = None) -> object:
    """Init watcher metrics server and structured file logging.

    Returns a WatcherMetrics instance (or None if prometheus_client is unavailable).
    """
    setup_logging()
    setup_cluster_logging(
        logger=logging.getLogger("smoltorrent.watcher"),
        component="server",
        hostname=hostname,
        log_dir=log_dir,
        algorithm="syncps",
        arch="smoltorrent",
    )
    if not HAS_PROM:
        logger.warning("[obs] prometheus_client not available — watcher metrics skipped")
        return None
    from utils.prometheus_utils import init_watcher_metrics
    return init_watcher_metrics()
