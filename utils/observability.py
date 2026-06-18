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
from typing import TYPE_CHECKING, Optional

from utils.log_utils import setup_logging, setup_cluster_logging
from utils.prometheus_utils import HAS_PROM

if TYPE_CHECKING:
    from utils.prometheus_utils import WatcherMetrics, WorkerMetrics

logger = logging.getLogger(__name__)



def setup_worker(rank: int, hostname: str, log_dir: Optional[str] = None) -> "Optional[WorkerMetrics]":
    """Initialise worker metrics server and structured file logging.

    Args:
        rank:     Integer rank of this worker (used to choose the Prometheus port).
        hostname: Human-readable hostname for log file naming and context labels.
        log_dir:  Optional override for the cluster-log directory.

    Returns:
        A :class:`~utils.prometheus_utils.WorkerMetrics` instance, or ``None``
        if ``prometheus_client`` is not installed on this node.
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


def setup_api(hostname: Optional[str] = None, log_dir: Optional[str] = None) -> None:
    """Initialise structured file logging for the FastAPI process.

    Writes to ``api-server-<hostname>.log`` in the cluster-log directory so
    Loki can scrape API logs separately from the watcher.

    Args:
        hostname: Hostname used in the log file name (defaults to empty string).
        log_dir:  Optional override for the cluster-log directory.

    Returns:
        None.
    """
    setup_logging()
    setup_cluster_logging(
        logger=logging.getLogger("backend"),
        component="server",
        hostname=hostname,
        log_dir=log_dir,
        algorithm="api",
        arch="smoltorrent",
    )
    # Also capture uvicorn.error and fastapi loggers (skip uvicorn.access —
    # it uses a specialized AccessFormatter that breaks with a plain FileHandler)
    for name in ("uvicorn", "uvicorn.error", "fastapi"):
        setup_cluster_logging(
            logger=logging.getLogger(name),
            component="server",
            hostname=hostname,
            log_dir=log_dir,
            algorithm="api",
            arch="smoltorrent",
        )


def setup_watcher(hostname: Optional[str] = None, log_dir: Optional[str] = None) -> "Optional[WatcherMetrics]":
    """Initialise watcher metrics server and structured file logging.

    Args:
        hostname: Hostname used in the log file name (defaults to empty string).
        log_dir:  Optional override for the cluster-log directory.

    Returns:
        A :class:`~utils.prometheus_utils.WatcherMetrics` instance, or ``None``
        if ``prometheus_client`` is not installed.
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
