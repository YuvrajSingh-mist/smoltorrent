"""mDNS discovery via zeroconf — works on Mac and Linux (Pi workers).

Workers advertise ``_smoltorrent._tcp.local.`` when they start.
The master calls ``WorkerBrowser()`` to find them without needing
any hardcoded IPs.

Browsers are the listeners that run in the master to discover workers, and advertisers are the broadcasters that run in the workers to announce themselves.
"""

import socket
import threading
import time

from typing import Optional

from zeroconf import ServiceBrowser, ServiceInfo, ServiceListener, Zeroconf

from ._utils import get_logger, get_local_ip

log = get_logger("mdns")

SERVICE_TYPE = "_smoltorrent._tcp.local."


class WorkerAdvertiser:
    """Advertise this worker over mDNS so the master can find it.

    Usage (in worker.py after the TCP socket is bound)::

        adv = WorkerAdvertiser(rank=1, port=5001)
        # ... worker runs forever ...
        adv.close()

    Also usable as a context manager.
    """

    def __init__(self, rank: int, port: int, hostname: Optional[str] = None) -> None:
        """Register a ``_smoltorrent._tcp.local.`` service.

        Args:
            rank:     Integer rank of this worker (must match config).
            port:     TCP port the worker's shard listener is bound to.
            hostname: Human-readable hostname for the TXT record.
                      Defaults to :func:`socket.gethostname`.
        """
        host = hostname or socket.gethostname()
        ip = get_local_ip()
        self.zc = Zeroconf()
        self.info = ServiceInfo(
            SERVICE_TYPE,
            f"smoltorrent-rank-{rank}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties={
                b"rank": str(rank).encode(),
                b"hostname": host.encode(),
            },
        )
        self.zc.register_service(self.info, allow_name_change=True)
        log.info(
            "[mdns] worker advertised: rank=%d host=%s ip=%s port=%d",
            rank, host, ip, port,
        )

    def close(self) -> None:
        """Unregister the mDNS service and release the Zeroconf handle."""
        self.zc.unregister_service(self.info)
        self.zc.close()
        log.info("[mdns] worker advertisement removed")

    def __enter__(self) -> "WorkerAdvertiser":
        """Return self to support use as a context manager.

        Args:
            None.

        Returns:
            This :class:`WorkerAdvertiser` instance.
        """
        return self

    def __exit__(self, *_) -> None:
        """Unregister the mDNS service on context manager exit.

        Args:
            *_: Exception info (ignored).

        Returns:
            None.
        """
        self.close()


def WorkerBrowser(timeout: float = 10.0) -> list[dict]:
    """Browse for ``_smoltorrent._tcp.local.`` services for *timeout* seconds.

    Spawns a :class:`~zeroconf.ServiceBrowser` that listens for mDNS
    announcements, collects every worker it hears, then returns them
    sorted by rank.  Equivalent to :class:`MasterBrowser` but for workers.

    Args:
        timeout: Seconds to listen before returning (default 10).

    Returns:
        List of worker dicts sorted by rank, each containing:

        * ``"ip"`` (:class:`str`) — IPv4 address
        * ``"port"`` (:class:`int`) — TCP port
        * ``"rank"`` (:class:`int`) — worker rank
        * ``"hostname"`` (:class:`str`) — hostname from the TXT record
    """
    found: dict[int, dict] = {}
    lock = threading.Lock()

    class Listener(ServiceListener):
        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            """Record a newly discovered worker service.

            Args:
                zc:    Active :class:`~zeroconf.Zeroconf` instance.
                type_: Service type string.
                name:  Full service name.

            Returns:
                None.
            """
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
                    rank = int(props.get("rank") or -1)
                except ValueError:
                    log.error("[mdns] invalid rank in service %s: %s", name, props.get("rank"))
                    return
                with lock:
                    found[rank] = {
                        "ip": ip,
                        "port": info.port,
                        "rank": rank,
                        "hostname": props.get("hostname", ""),
                    }
                    log.info("[mdns] found worker: rank=%d host=%s ip=%s port=%d", rank, props.get("hostname", ""), ip, info.port)

        def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            """No-op: worker removals are not tracked during one-shot discovery.

            Args:
                zc:    Active :class:`~zeroconf.Zeroconf` instance.
                type_: Service type string.
                name:  Full service name.

            Returns:
                None.
            """
            pass

        def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            """Re-process an updated service announcement as if it were new.

            Args:
                zc:    Active :class:`~zeroconf.Zeroconf` instance.
                type_: Service type string.
                name:  Full service name.

            Returns:
                None.
            """
            self.add_service(zc, type_, name)

    zc = Zeroconf()
    log.info("[mdns] browsing for %s (timeout=%ss)...", SERVICE_TYPE, timeout)
    browser = ServiceBrowser(zc, SERVICE_TYPE, Listener())
    log.info("[mdns] ServiceBrowser started")
    time.sleep(timeout)
    browser.cancel()
    zc.close()
    log.info("[mdns] ServiceBrowser stopped — processing results...")
    result = sorted(found.values(), key=lambda x: x["rank"])
    log.info("[mdns] discovery finished — %d worker(s) found", len(result))
    return result


MASTER_SERVICE_TYPE = "_smolt-master._tcp.local."
REGISTRATION_PORT = 5999


class MasterAdvertiser:
    """Advertise this node as a smoltorrent master over mDNS.

    Workers running ``python main.py join`` will see it in their JoinApp TUI.
    """

    def __init__(self, expected_workers: Optional[int] = None) -> None:
        """Advertise this node as a smoltorrent master.

        Args:
            expected_workers: Number of workers the master will wait for
                              before launching.  Stored in the TXT record
                              so the JoinApp TUI can show progress.
        """
        self.zc = Zeroconf()
        hostname = socket.gethostname()
        ip = get_local_ip()
        self.info = ServiceInfo(
            MASTER_SERVICE_TYPE,
            f"smoltorrent-{hostname}.{MASTER_SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=REGISTRATION_PORT,
            properties={
                b"hostname": hostname.encode(),
                # b"expected": str(expected_workers).encode(),
                b"current": b"0",
                b"started": str(time.time()).encode(),
            },
        )
        self.zc.register_service(self.info, allow_name_change=True)
        log.info(
            "[mdns] master advertised: host=%s ip=%s",
            hostname, ip
        )
        # log.info(
        #     "[mdns] master advertised: host=%s ip=%s expected_workers=%d",
        #     hostname, ip, expected_workers,
        # )

    def update_current(self, count: int) -> None:
        """Update the ``current`` worker count in the live mDNS TXT record.

        Args:
            count: Number of workers that have registered so far.

        Returns:
            None.
        """
        self.info.properties[b"current"] = str(count).encode()
        self.zc.update_service(self.info)

    def close(self) -> None:
        """Unregister the master mDNS service and release the Zeroconf handle.

        Args:
            None.

        Returns:
            None.
        """
        self.zc.unregister_service(self.info)
        self.zc.close()
        log.info("[mdns] master advertisement removed")


class MasterBrowser(ServiceListener):
    """Live mDNS browser for smoltorrent masters.

    Returns data in the format ``JoinApp`` expects via ``get_clusters()``.
    """

    def __init__(self) -> None:
        """Start browsing for smoltorrent master mDNS advertisements.

        Args:
            None.

        Returns:
            None.
        """
        self.masters: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.zc = Zeroconf()
        ServiceBrowser(self.zc, MASTER_SERVICE_TYPE, self)
        log.info("[mdns] MasterBrowser started — listening for %s", MASTER_SERVICE_TYPE)

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Record a newly discovered master service in the internal registry.

        Args:
            zc:    Active :class:`~zeroconf.Zeroconf` instance.
            type_: Service type string.
            name:  Full service name (used as the registry key).

        Returns:
            None.
        """
        info = zc.get_service_info(type_, name)
        if not info or not info.addresses:
            return
        props = {
            k.decode() if isinstance(k, bytes) else k: v.decode()
            if isinstance(v, bytes)
            else v
            for k, v in info.properties.items()
        }
        hostname = props.get("hostname", name)
        with self.lock:
            self.masters[name] = {
                "name": f"smoltorrent @ {hostname}",
                "uid": name,
                "hostname": hostname,
                "ip": socket.inet_ntoa(info.addresses[0]),
                "port": info.port,
                # "expected": int(props.get("expected", 1)),
                "current": int(props.get("current") or 0),
                "started": props.get("started", str(time.time())),
            }
        log.info("[mdns] MasterBrowser found master: %s (%s)", hostname, socket.inet_ntoa(info.addresses[0]))

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Re-process an updated master service announcement as if it were new.

        Args:
            zc:    Active :class:`~zeroconf.Zeroconf` instance.
            type_: Service type string.
            name:  Full service name.

        Returns:
            None.
        """
        self.add_service(zc, type_, name)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Remove a departed master from the internal registry.

        Args:
            zc:    Active :class:`~zeroconf.Zeroconf` instance (unused).
            type_: Service type string (unused).
            name:  Full service name used as the registry key.

        Returns:
            None.
        """
        with self.lock:
            self.masters.pop(name, None)
        log.info("[mdns] MasterBrowser lost master: %s", name)

    def get_clusters(self) -> list[dict]:
        """Return a snapshot of every visible master.

        Returns:
            List of master dicts, each with keys ``name``, ``uid``,
            ``hostname``, ``ip``, ``port``, ``current``,
            ``started``.
        """
        with self.lock:
            return list(self.masters.values())

    def close(self) -> None:
        """Stop browsing and release the Zeroconf handle."""
        self.zc.close()
        log.info("[mdns] MasterBrowser closed")



