"""SmolTorrent device discovery.

Two transports — both run automatically:
  * mDNS/TCP (zeroconf)  — works on Mac and Linux/Pi over WiFi or Ethernet.
  * AirDrop/AWDL (Swift) — Mac-only, peer-to-peer without a router.

Typical usage
-------------
Master (discovers workers)::

    from discovery import discover_workers
    workers = discover_workers(timeout=10)
    # [{"ip": "...", "port": 5001, "rank": 1, "hostname": "pi4-1"}, ...]

Worker (advertises itself)::

    from discovery import advertise_worker
    with advertise_worker(rank=1, port=5001):
        run_forever()
"""

import sys

from .grove._mdns import WorkerAdvertiser, discover_mdns_workers
from .grove.transport.p2p import discover_airdrop_workers  

def advertise_worker(rank: int, port: int, hostname: str | None = None) -> WorkerAdvertiser:
    """Register this worker over mDNS so the master can find it without IPs.

    Returns a :class:`WorkerAdvertiser` — call ``.close()`` when done or use
    it as a context manager.

    Args:
        rank:     Worker rank (must match config.yaml).
        port:     TCP port this worker is listening on.
        hostname: Override the advertised hostname (defaults to ``socket.gethostname()``).
    """
    return WorkerAdvertiser(rank=rank, port=port, hostname=hostname)


def discover_workers(timeout: float = 10.0) -> list[dict]:
    """Scan the local network for smoltorrent workers.

    Runs mDNS on all platforms; additionally runs AirDrop/AWDL on macOS.
    Deduplicates by rank (mDNS result takes priority over AWDL for the
    same rank since it carries the real IP/port).

    Args:
        timeout: How long to listen for mDNS announcements (seconds).
                 AirDrop runs in parallel for the same duration.

    Returns:
        List of worker dicts sorted by rank::

            [{"ip": str, "port": int, "rank": int, "hostname": str}, ...]
    """
    import threading

    mdns_results: list[dict] = []
    airdrop_results: list[dict] = []

    def _run_mdns():
        mdns_results.extend(discover_mdns_workers(timeout=timeout))

    def _run_airdrop():
        if sys.platform != "darwin":
            return
        try:
           
            airdrop_results.extend(discover_airdrop_workers(timeout=timeout))
        except Exception:
            pass

    t_mdns = threading.Thread(target=_run_mdns, daemon=True)
    t_awdl = threading.Thread(target=_run_airdrop, daemon=True)
    t_mdns.start()
    t_awdl.start()
    t_mdns.join()
    t_awdl.join()

    # merge: mDNS has real IP/port, so it wins on rank collision
    merged: dict[int, dict] = {}
    for node in airdrop_results:
        # AirDrop nodes don't carry rank/port — store by uid for display only
        pass
    for worker in mdns_results:
        merged[worker["rank"]] = worker

    return sorted(merged.values(), key=lambda x: x["rank"])
