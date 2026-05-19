"""Tests for device discovery — mDNS (local Mac) and live /discover endpoint.

Markers:
  (none)        — local mDNS loopback: advertise a fake worker on this Mac,
                  discover it, confirm it appears. No cluster needed.
  api           — live /discover endpoint: requires API running + Pi workers up.
"""

import socket
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from discovery import discover_workers
from discovery._mdns import WorkerAdvertiser, discover_mdns_workers

API_BASE = "http://localhost:8000"
_FAKE_RANK = 99
_FAKE_PORT = 19099


# ---------------------------------------------------------------------------
# Local Mac mDNS loopback (no cluster required)
# ---------------------------------------------------------------------------


class TestMdnsLoopback:
    """Start a fake WorkerAdvertiser on this Mac, discover it locally.

    Confirms zeroconf works end-to-end on macOS without any Pis or API.
    """

    def test_advertise_and_discover(self):
        adv = WorkerAdvertiser(rank=_FAKE_RANK, port=_FAKE_PORT)
        try:
            # Give mDNS a moment to register before scanning
            time.sleep(1.0)
            found = discover_mdns_workers(timeout=6.0)
        finally:
            adv.close()

        ranks = [w["rank"] for w in found]
        assert _FAKE_RANK in ranks, (
            f"Fake rank {_FAKE_RANK} not found in mDNS scan — got ranks: {ranks}"
        )

    def test_advertised_port_matches(self):
        with WorkerAdvertiser(rank=_FAKE_RANK, port=_FAKE_PORT):
            time.sleep(1.0)
            found = discover_mdns_workers(timeout=6.0)

        worker = next((w for w in found if w["rank"] == _FAKE_RANK), None)
        assert worker is not None, f"Rank {_FAKE_RANK} not found"
        assert worker["port"] == _FAKE_PORT

    def test_advertised_hostname_matches(self):
        with WorkerAdvertiser(rank=_FAKE_RANK, port=_FAKE_PORT):
            time.sleep(1.0)
            found = discover_mdns_workers(timeout=6.0)

        worker = next((w for w in found if w["rank"] == _FAKE_RANK), None)
        assert worker is not None
        assert worker["hostname"] == socket.gethostname()

    def test_advertised_ip_is_not_loopback(self):
        with WorkerAdvertiser(rank=_FAKE_RANK, port=_FAKE_PORT):
            time.sleep(1.0)
            found = discover_mdns_workers(timeout=6.0)

        worker = next((w for w in found if w["rank"] == _FAKE_RANK), None)
        assert worker is not None
        assert worker["ip"] != "127.0.0.1", "Expected a real LAN IP, not loopback"

    def test_context_manager_closes_cleanly(self):
        with WorkerAdvertiser(rank=_FAKE_RANK, port=_FAKE_PORT):
            time.sleep(0.5)
        # After __exit__, the service should be unregistered — no exception = pass

    def test_discover_workers_public_api(self):
        """discover_workers() from the public __init__ API should also find it."""
        with WorkerAdvertiser(rank=_FAKE_RANK, port=_FAKE_PORT):
            time.sleep(1.0)
            found = discover_workers(timeout=6.0)

        assert any(w["rank"] == _FAKE_RANK for w in found)

    def test_multiple_advertisers_all_found(self):
        """Two fake workers advertised at the same time are both discovered."""
        adv_a = WorkerAdvertiser(rank=97, port=19097)
        adv_b = WorkerAdvertiser(rank=98, port=19098)
        try:
            time.sleep(1.0)
            found = discover_mdns_workers(timeout=6.0)
        finally:
            adv_a.close()
            adv_b.close()

        ranks = {w["rank"] for w in found}
        assert 97 in ranks, f"rank 97 not found — got {ranks}"
        assert 98 in ranks, f"rank 98 not found — got {ranks}"


# ---------------------------------------------------------------------------
# Live /discover endpoint (requires API + Pi workers running)
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestDiscoverEndpoint:
    """GET /discover: hits the real API and expects all 4 Pi workers."""

    def test_returns_200(self):
        resp = httpx.get(f"{API_BASE}/discover", params={"timeout": 10}, timeout=20)
        assert resp.status_code == 200

    def test_response_has_workers_key(self):
        resp = httpx.get(f"{API_BASE}/discover", params={"timeout": 10}, timeout=20)
        assert "workers" in resp.json()

    def test_all_four_workers_found(self):
        resp = httpx.get(f"{API_BASE}/discover", params={"timeout": 10}, timeout=20)
        workers = resp.json()["workers"]
        ranks = {w["rank"] for w in workers}
        assert ranks == {1, 2, 3, 4}, f"Expected ranks {{1,2,3,4}}, got {ranks}"

    def test_workers_have_required_fields(self):
        resp = httpx.get(f"{API_BASE}/discover", params={"timeout": 10}, timeout=20)
        for w in resp.json()["workers"]:
            assert "ip" in w and w["ip"], f"Missing ip in {w}"
            assert "port" in w and w["port"] > 0, f"Missing/invalid port in {w}"
            assert "rank" in w, f"Missing rank in {w}"
            assert "hostname" in w and w["hostname"], f"Missing hostname in {w}"

    def test_workers_sorted_by_rank(self):
        resp = httpx.get(f"{API_BASE}/discover", params={"timeout": 10}, timeout=20)
        workers = resp.json()["workers"]
        ranks = [w["rank"] for w in workers]
        assert ranks == sorted(ranks), f"Workers not sorted by rank: {ranks}"

    def test_ips_are_not_loopback(self):
        resp = httpx.get(f"{API_BASE}/discover", params={"timeout": 10}, timeout=20)
        for w in resp.json()["workers"]:
            assert not w["ip"].startswith("127."), f"Worker {w['rank']} has loopback IP"

    def test_short_timeout_still_returns_json(self):
        """A very short timeout returns an empty or partial list — not an error."""
        resp = httpx.get(f"{API_BASE}/discover", params={"timeout": 0.1}, timeout=10)
        assert resp.status_code == 200
        assert "workers" in resp.json()
