"""AirDrop/AWDL node discovery via the Swift P2P helper sidecar.

Only the discovery surface is kept here — the full P2PTransport mesh
(used for grove's training communication) has been removed since
smoltorrent only needs to *find* nodes, not exchange tensors over AWDL.
"""

import os
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Generator

from .._utils import get_logger, setup_grove_logging

log = get_logger("p2p")


def read_line(sock: socket.socket) -> str:
    """Read bytes from *sock* until a newline, then return the decoded string.

    Reads one byte at a time — acceptable because the discovery protocol
    exchanges fewer than 10 lines per session.  For higher-throughput
    protocols prefer a length-prefixed binary frame.

    Args:
        sock: A connected :class:`socket.socket` (UDS or TCP).

    Returns:
        The decoded line with the trailing ``\\n`` stripped, or an empty
        string if the connection closed before any data arrived.
    """
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            break
        buf += chunk
    return buf.decode().strip()


def parse_node(text: str) -> dict | None:
    """Parse a ``found <name> <uid> ...`` line into a node dict.

    Args:
        text: A raw line from the Swift helper, e.g.
              ``"found MacBook abc123 4 train.py"``.

    Returns:
        A dict with keys ``name``, ``uid``, ``hostname``, ``started``,
        or ``None`` if the line does not have enough fields.
    """
    parts = text.split(None, 4)
    if len(parts) >= 3:
        return {
            "name": parts[1],
            "uid": parts[2],
            "hostname": "awdl-peer",
            "started": str(time.time()),
        }
    return None


def log_stderr(proc: subprocess.Popen, label: str) -> None:
    """Read stderr from *proc* line-by-line and route into the grove log.

    Runs as a daemon thread — never returns on its own; terminates when
    the subprocess exits and closes stderr.  Every non-empty line is
    logged at ``DEBUG`` level with a ``[swift <label>]`` prefix.

    Args:
        proc:  A :class:`~subprocess.Popen` whose ``stderr`` is a pipe.
        label: Human-readable tag for log lines (e.g. ``"discover"``).
    """
    assert proc.stderr is not None
    for raw_line in proc.stderr:
        line = raw_line.decode(errors="replace").strip()
        if line:
            log.debug("[swift %s] %s", label, line)


@contextmanager
def swift_discover(label: str) -> Generator[socket.socket, None, None]:
    """Launch the Swift helper in discover mode and yield a connected UDS socket.

    Context manager that handles the full lifecycle: compilation check,
    subprocess launch, stderr capture, UDS connection with 30 s retry,
    ``"ready"`` consumption, and cleanup (``terminate`` + ``unlink``).

    On non-Mac platforms (no Swift compiler) the generator yields
    ``None`` immediately without raising.

    Args:
        label: Short tag used in log messages and the UDS path, e.g.
               ``"discover"`` or ``"live"``.

    Yields:
        A connected :class:`socket.socket` over ``AF_UNIX``, or
        ``None`` if the helper could not be started.
    """
    setup_grove_logging()

    try:
        from ..swift.compile import ensure_compiled

        helper_path = ensure_compiled()
    except RuntimeError:
        log.info("[p2p] %s skipped — Swift compiler unavailable", label)
        yield None
        return

    log.info("[p2p] %s starting via %s", label, helper_path)

    ctrl_path = f"/tmp/smoltorrent_{label}_{os.getpid()}.sock"
    proc = subprocess.Popen(
        [str(helper_path), "discover", ctrl_path], stderr=subprocess.PIPE
    )
    threading.Thread(target=log_stderr, args=(proc, label), daemon=True).start()

    sock = None
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(ctrl_path)
            break
        except (FileNotFoundError, ConnectionRefusedError) as e:
            log.warning("[p2p] %s connect error: %s — retrying...", label, e)
            time.sleep(0.05)

    if sock is None:
        log.warning("[p2p] %s — Swift helper didn't start within 30s", label)
        proc.terminate()
        yield None
        return

    read_line(sock)  # consume "ready" line
    log.debug("[p2p] %s helper connected — listening for announcements", label)

    try:
        yield sock
    finally:
        sock.close()
        proc.terminate()
        proc.wait(timeout=3)
        if os.path.exists(ctrl_path):
            os.unlink(ctrl_path)
        log.info("[p2p] %s helper terminated and cleaned up", label)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_airdrop_workers(timeout: float = 10.0) -> list[dict]:
    """Scan for smoltorrent nodes over AirDrop/AWDL (macOS only).

    Launches the Swift helper in one-shot discover mode, collects every
    ``found`` announcement for *timeout* seconds, then terminates the
    helper and returns the results.  Silently returns an empty list on
    non-Mac platforms or when Swift compilation fails.

    Args:
        timeout: Seconds to listen for AWDL announcements (default 10).

    Returns:
        List of node dicts, each with keys ``name`` (:class:`str`),
        ``uid`` (:class:`str`), ``hostname`` (:class:`str` — always
        ``"awdl-peer"``), ``started`` (:class:`str` — ISO timestamp).
    """
    nodes: list[dict] = []

    with swift_discover("discover") as sock:
        if sock is None:
            return []

        sock.settimeout(1.0)
        end_time = time.monotonic() + timeout

        while time.monotonic() < end_time:
            try:
                text = read_line(sock)
                if not text:
                    break
                if text.startswith("found "):
                    node = parse_node(text)
                    if node:
                        nodes.append(node)
                        log.info("[p2p] found node: %s (uid=%s)", node["name"], node["uid"])
            except TimeoutError:
                continue

    log.info("[p2p] discovery finished — %d node(s) found", len(nodes))
    return nodes
