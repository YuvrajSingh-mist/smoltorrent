"""Shared utilities."""

import logging
import os
import socket
from pathlib import Path
from typing import Optional

from ._types import DEFAULT_SOCK_BUF_SIZE

log = logging.getLogger("grove.utils")

# ---------------------------------------------------------------------------
# Grove file logging — writes to logging/grove/ (sibling to logging/cluster-logs/)
# ---------------------------------------------------------------------------

FILE_LOGGING_SETUP = False
GROVE_LOG_DIR: Optional[Path] = None


def setup_grove_logging(
    level: int = logging.INFO,
    *,
    log_dir: Optional[str] = None,
) -> Path:
    """Add a file handler to the ``grove`` parent logger.

    All ``grove.*`` child loggers (``grove.utils``, ``grove.swift``,
    ``grove.p2p``, ``grove.mdns``, etc.) automatically inherit this
    handler — console output continues unchanged.

    Safe to call multiple times (idempotent).  Call once from the main
    entry point before any grove submodule is used.

    Returns:
        Absolute path to the log file.
    """
    global FILE_LOGGING_SETUP, GROVE_LOG_DIR

    if FILE_LOGGING_SETUP:
        assert GROVE_LOG_DIR is not None
        return GROVE_LOG_DIR

    if log_dir:
        grove_dir = Path(log_dir)
    else:
        # Project root is 4 levels up from this file:
        #   discovery/grove/_utils.py  →  project root
        grove_dir = Path(__file__).resolve().parents[3] / "logging" / "grove"

    grove_dir.mkdir(parents=True, exist_ok=True)
    log_path = grove_dir / "discovery.log"

    parent = logging.getLogger("grove")
    parent.setLevel(level)

    # Idempotency: skip if this exact file handler is already attached
    if any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
        for h in parent.handlers
    ):
        FILE_LOGGING_SETUP = True
        GROVE_LOG_DIR = log_path
        return log_path

    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  [grove]  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    parent.addHandler(fh)

    FILE_LOGGING_SETUP = True
    GROVE_LOG_DIR = log_path
    log.info("Grove file logging ready: %s", log_path)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Create (or retrieve) a ``grove.<name>`` child logger.

    The returned logger always has a ``StreamHandler`` writing to stderr
    with the format ``[grove <name>] LEVEL: message``.  If the logger
    already exists it is returned unchanged — handlers are never duplicated.

    Args:
        name: Short name for this submodule, e.g. ``"p2p"``, ``"mdns"``, ``"tui"``.

    Returns:
        A :class:`logging.Logger` whose name is ``grove.<name>``.
    """
    logger = logging.getLogger(f"grove.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[grove %(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_local_ip() -> str:
    """Return the machine's primary routable IPv4 address.

    Uses a zero-traffic UDP trick: ``connect()`` on a UDP socket to
    ``8.8.8.8:80`` forces the kernel to choose the local interface that
    can reach the outside world, then ``getsockname()`` reads the
    assigned address.  No packets are ever sent.

    Returns:
        IPv4 address as a string, e.g. ``"192.168.1.42"``.
        Falls back to ``"127.0.0.1"`` when no route exists.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def recvall(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *sock*, blocking until all have arrived.

    Unlike :meth:`socket.recv`, which may return fewer bytes than
    requested, this function loops until the full *n* bytes are in hand
    or the connection closes.

    Args:
        sock: A connected, blocking :class:`socket.socket`.
        n:    Exact number of bytes to read.

    Returns:
        A :class:`bytes` object of length *n*.

    Raises:
        ConnectionError: If the peer closes the connection before *n*
            bytes have been received.
    """
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        nbytes = sock.recv_into(view[pos:])
        if nbytes == 0:
            raise ConnectionError("Connection closed during recv")
        pos += nbytes
    return bytes(buf)


def configure_socket(sock: socket.socket, *, tcp: bool = False) -> None:
    """Apply performance-oriented socket options to *sock*.

    For TCP sockets enables ``TCP_NODELAY`` (disable Nagle), keep-alive
    probes, and large send/receive buffers.  For all sockets, sets
    ``SO_SNDBUF`` and ``SO_RCVBUF`` to :data:`DEFAULT_SOCK_BUF_SIZE`
    (32 MiB).  Failures are logged at DEBUG level but never raised.

    Args:
        sock: A :class:`socket.socket` that has not yet been connected.
        tcp:  Set ``True`` for TCP-specific options (Nagle, keepalive).
    """
    if tcp:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, 0x10, 5)  # TCP_KEEPIDLE: 5s
            sock.setsockopt(socket.IPPROTO_TCP, 0x101, 2)  # TCP_KEEPINTVL: 2s
            sock.setsockopt(socket.IPPROTO_TCP, 0x102, 3)  # TCP_KEEPCNT: 3 probes
        except OSError as e:
            log.debug("[utils] failed to set TCP keepalive: %s", e)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, DEFAULT_SOCK_BUF_SIZE)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, DEFAULT_SOCK_BUF_SIZE)
    except OSError as e:
        log.debug("[utils] failed to set socket buffer size: %s", e)
