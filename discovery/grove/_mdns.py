"""mDNS discovery via zeroconf — works on Mac and Linux (Pi workers).

Workers advertise ``_smoltorrent._tcp.local.`` when they start.
The master calls ``discover_mdns_workers()`` to find them without needing
any hardcoded IPs.
"""

import socket
import threading
import time

from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf

_SERVICE_TYPE = "_smoltorrent._tcp.local."


class WorkerAdvertiser:
    """Advertise this worker over mDNS so the master can find it.

    Usage (in worker.py after the TCP socket is bound)::

        adv = WorkerAdvertiser(rank=1, port=5001)
        # ... worker runs forever ...
        adv.close()

    Also usable as a context manager.
    """

    def __init__(self, rank: int, port: int, hostname: str | None = None) -> None:
        host = hostname or socket.gethostname()
        ip = _get_local_ip()
        self._zc = Zeroconf()
        self._info = ServiceInfo(
            _SERVICE_TYPE,
            f"smoltorrent-rank-{rank}.{_SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties={
                b"rank": str(rank).encode(),
                b"hostname": host.encode(),
            },
        )
        self._zc.register_service(self._info, allow_name_change=True)

    def close(self) -> None:
        self._zc.unregister_service(self._info)
        self._zc.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def discover_mdns_workers(timeout: float = 10.0) -> list[dict]:
    """Browse for ``_smoltorrent._tcp.local.`` services for ``timeout`` seconds.

    Returns:
        List of worker dicts sorted by rank::

            [{"ip": "192.168.1.x", "port": 5001, "rank": 1, "hostname": "pi4-1"}, ...]
    """
    found: dict[int, dict] = {}
    lock = threading.Lock()

    class _Listener:
        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            info = zc.get_service_info(type_, name)
            if info and info.addresses:
                ip = socket.inet_ntoa(info.addresses[0])
                props = {
                    k.decode() if isinstance(k, bytes) else k: v.decode()
                    if isinstance(v, bytes)
                    else v
                    for k, v in info.properties.items()
                }
                try:
                    rank = int(props.get("rank", -1))
                except ValueError:
                    return
                with lock:
                    found[rank] = {
                        "ip": ip,
                        "port": info.port,
                        "rank": rank,
                        "hostname": props.get("hostname", ""),
                    }

        def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            pass

        def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            self.add_service(zc, type_, name)

    zc = Zeroconf()
    browser = ServiceBrowser(zc, _SERVICE_TYPE, _Listener())
    time.sleep(timeout)
    browser.cancel()
    zc.close()

    return sorted(found.values(), key=lambda x: x["rank"])


_MASTER_SERVICE_TYPE = "_smolt-master._tcp.local."
_REGISTRATION_PORT = 5999


class MasterAdvertiser:
    """Advertise this node as a smoltorrent master over mDNS.

    Workers running ``python main.py join`` will see it in their JoinApp TUI.
    """

    def __init__(self, expected_workers: int) -> None:
        self._zc = Zeroconf()
        hostname = socket.gethostname()
        ip = _get_local_ip()
        self._info = ServiceInfo(
            _MASTER_SERVICE_TYPE,
            f"smoltorrent-{hostname}.{_MASTER_SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=_REGISTRATION_PORT,
            properties={
                b"hostname": hostname.encode(),
                b"expected": str(expected_workers).encode(),
                b"current": b"0",
                b"started": str(time.time()).encode(),
            },
        )
        self._zc.register_service(self._info, allow_name_change=True)

    def close(self) -> None:
        self._zc.unregister_service(self._info)
        self._zc.close()


class MasterBrowser:
    """Live mDNS browser for smoltorrent masters.

    Returns data in the format ``JoinApp`` expects via ``get_clusters()``.
    """

    def __init__(self) -> None:
        self._masters: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._zc = Zeroconf()
        ServiceBrowser(self._zc, _MASTER_SERVICE_TYPE, self)

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if not info or not info.addresses:
            return
        props = {
            k.decode() if isinstance(k, bytes) else k: v.decode()
            if isinstance(v, bytes)
            else v
            for k, v in info.properties.items()
        }
        with self._lock:
            self._masters[name] = {
                "name": f"smoltorrent @ {props.get('hostname', name)}",
                "uid": name,
                "hostname": props.get("hostname", name),
                "ip": socket.inet_ntoa(info.addresses[0]),
                "port": info.port,
                "expected": int(props.get("expected", 1)),
                "current": int(props.get("current", 0)),
                "started": props.get("started", str(time.time())),
            }

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self.add_service(zc, type_, name)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        with self._lock:
            self._masters.pop(name, None)

    def get_clusters(self) -> list[dict]:
        with self._lock:
            return list(self._masters.values())

    def close(self) -> None:
        self._zc.close()


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"
