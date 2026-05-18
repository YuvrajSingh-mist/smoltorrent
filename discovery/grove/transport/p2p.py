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


def _read_line(sock: socket.socket) -> str:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            break
        buf += chunk
    return buf.decode().strip()


def discover_airdrop_workers(timeout: float = 10.0) -> list[dict]:
    """Scan for smoltorrent nodes over AirDrop/AWDL (Mac only).

    Launches the Swift helper in discover mode, collects announcements
    for ``timeout`` seconds, then returns the found nodes.

    Returns:
        List of dicts with keys ``name``, ``uid``, ``hostname``, ``started``.
        Empty list on non-Mac platforms or if swiftc is unavailable.
    """
    try:
        from ..swift.compile import ensure_compiled
        helper_path = ensure_compiled()
    except RuntimeError:
        return []

    ctrl_path = f"/tmp/smoltorrent_discover_{os.getpid()}.sock"
    proc = subprocess.Popen([str(helper_path), "discover", ctrl_path], stderr=subprocess.PIPE)

    sock = None
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(ctrl_path)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.05)

    if sock is None:
        proc.terminate()
        return []

    _read_line(sock)  # consume "ready" line

    nodes: list[dict] = []
    sock.settimeout(1.0)
    end_time = time.monotonic() + timeout

    while time.monotonic() < end_time:
        try:
            text = _read_line(sock)
            if not text:
                break
            if text.startswith("found "):
                parts = text.split(None, 4)
                if len(parts) >= 5:
                    nodes.append({"name": parts[1], "uid": parts[2], "hostname": "awdl-peer", "started": str(time.time())})
        except TimeoutError:
            continue

    sock.close()
    proc.terminate()
    proc.wait(timeout=3)
    if os.path.exists(ctrl_path):
        os.unlink(ctrl_path)

    return nodes


class AirdropBrowser:
    """Live browser for AirDrop/AWDL smoltorrent nodes (Mac only).

    Runs the Swift helper in the background and maintains a live set of
    visible nodes. Call :meth:`get_nodes` at any time; :meth:`close` when done.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._running = True

        try:
            from ..swift.compile import ensure_compiled
            helper_path = ensure_compiled()
        except RuntimeError:
            self._proc = None
            return

        ctrl_path = f"/tmp/smoltorrent_live_{os.getpid()}.sock"
        self._ctrl_path = ctrl_path
        self._proc = subprocess.Popen([str(helper_path), "discover", ctrl_path], stderr=subprocess.PIPE)

        deadline = time.monotonic() + 30.0
        self._sock: socket.socket | None = None
        while time.monotonic() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(ctrl_path)
                self._sock = s
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.05)

        if self._sock is None:
            self._proc.terminate()
            self._proc = None
            return

        _read_line(self._sock)
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self) -> None:
        self._sock.settimeout(1.0)
        while self._running:
            try:
                text = _read_line(self._sock)
                if not text:
                    return
                if text.startswith("found "):
                    parts = text.split(None, 4)
                    if len(parts) >= 5:
                        uid = parts[2]
                        with self._lock:
                            self._nodes[uid] = {"name": parts[1], "uid": uid, "hostname": "awdl-peer", "started": str(time.time())}
                elif text.startswith("lost "):
                    uid = text.split()[1]
                    with self._lock:
                        self._nodes.pop(uid, None)
            except TimeoutError:
                continue
            except (ConnectionError, OSError):
                break

    def get_nodes(self) -> list[dict]:
        with self._lock:
            return list(self._nodes.values())

    def close(self) -> None:
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=3)
            if os.path.exists(self._ctrl_path):
                os.unlink(self._ctrl_path)
