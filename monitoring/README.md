# SmolTorrent Monitoring

Prometheus + Grafana + Loki in Docker on the Mac. All logs from master (API, watcher) and all 4 Pi workers stream into one Grafana view.

---

## Quick start

```bash
# Start everything
bash scripts/launch_monitoring.sh

# Stop
bash scripts/launch_monitoring.sh --down
```

Grafana: http://127.0.0.1:3000 — `admin` / `smoltorrent`
Dashboard: http://127.0.0.1:3000/d/smoltorrent/smoltorrent

Data is persisted in Docker volumes (`prometheus-data`, `loki-data`, `grafana-data`) — dashboards and history survive restarts.

---

## All commands

| Command | What it does |
|---|---|
| `bash scripts/launch_monitoring.sh` | Preflight → start colima → bring stack up → wait until healthy |
| `bash scripts/launch_monitoring.sh --down` | Stop all containers (volumes preserved) |
| `bash scripts/launch_monitoring.sh --daemons` | Register LaunchDaemon — stack auto-starts on every Mac boot |
| `bash scripts/launch_monitoring.sh --install-pi-promtail` | Install Promtail on all 4 Pi workers via SSH |
| `bash scripts/launch_monitoring.sh --install-pi-promtail --workers 1,3` | Specific ranks only |

---

## Preflight checks

The script checks all of the following before doing anything. Failures print clearly and abort.

| Check | Auto-fix |
|---|---|
| macOS (monitoring runs on the Mac master only) | — |
| `colima` installed | — |
| `docker` installed | — |
| `docker-compose` installed | installs via `brew` |
| `python3` + `yaml` importable | — |
| `configs/config.yaml` exists | — |
| `monitoring/docker-compose.yml` exists | — |
| Prometheus / Loki / Promtail sub-configs present | — |
| Log dir exists and is writable | creates it |
| SSH key exists (only for `--install-pi-promtail`) | — |

---

## Auto-start on boot (macOS)

Run once from your terminal:

```bash
bash scripts/launch_monitoring.sh --daemons
```

This writes a startup script to `/usr/local/bin/smoltorrent_monitoring_startup.sh` and registers a LaunchDaemon at `/Library/LaunchDaemons/com.smoltorrent.monitoring.plist`. On every boot it waits for colima to be ready, then runs `docker-compose up -d`.

Individual container crashes (e.g. Loki OOM) are handled automatically by Docker's own `restart: unless-stopped` policy — launchd only needs to fire once at boot.

```bash
# Verify it's registered
sudo launchctl print system/com.smoltorrent.monitoring

# Watch the boot log
tail -f /tmp/smoltorrent-monitoring-startup.log

# Uninstall
sudo launchctl bootout system/com.smoltorrent.monitoring
sudo rm /Library/LaunchDaemons/com.smoltorrent.monitoring.plist
sudo rm /usr/local/bin/smoltorrent_monitoring_startup.sh
```

> **macOS 26 Tahoe note**: `LaunchAgents` and `launchctl load` are broken in Tahoe. This uses `/Library/LaunchDaemons` + `launchctl bootstrap system`, which is the only approach that works. The plist sets `UserName` so it runs as your user, not root.

---

## Pi worker logs

Pi workers write logs to `/tmp/smolcluster-logs/` on each Pi. Install Promtail there once to ship them to Loki:

```bash
# All 4 workers
bash scripts/launch_monitoring.sh --install-pi-promtail

# Specific ranks only
bash scripts/launch_monitoring.sh --install-pi-promtail --workers 1,3
```

This SSHes into each Pi, downloads Promtail v2.9.10 (last version with standalone arm64 binary), installs it as a systemd service (`smoltorrent-promtail`), and starts it. Promtail auto-restarts on Pi reboot via systemd. After that, Pi logs appear in Grafana automatically.

---

## Viewing logs in Grafana

Grafana → **Explore** → select **Loki** → run a query:

| Query | What you see |
|---|---|
| `{job="smoltorrent"}` | All nodes in one stream |
| `{job="smoltorrent", node="pi4-2"}` | One Pi only |
| `{job="smoltorrent", node="master"}` | Mac API + watcher only |
| `{job="smoltorrent", level="ERROR"}` | Errors across all nodes |
| `{job="smoltorrent"} \|= "rank 3"` | Lines mentioning rank 3 |

The **Cluster Logs** panel on the dashboard already runs `{job="smoltorrent"}`.

---

## What's in the dashboard

| Panel | What to look for |
|---|---|
| **Network Throughput MB/s** | Curve that starts fast and decelerates = O(n²) memory copying bug |
| **Transfer Duration p95** | Rising p95 = slow worker, flaky network, or SD card degrading |
| **Transfer Errors by Worker** | A rank that keeps erroring = dead port or Pi rebooted without systemd |
| **Operations** | Total store + gather completions since API started |
| **Avg Buffer Size** | Sudden spike = unusually large checkpoint being transferred |
| **Cluster Logs** | Unified log stream — filter by node or level |

---

## Metrics reference

Exposed at `http://localhost:8000/metrics/` by `backend/api.py` and `networking/send_receive.py`.

| Metric | Type | Description |
|---|---|---|
| `smoltorrent_bytes_sent_total` | Counter | Raw bytes sent over TCP |
| `smoltorrent_bytes_recv_total` | Counter | Raw bytes received over TCP |
| `smoltorrent_send_duration_seconds` | Histogram | Duration of each TCP send |
| `smoltorrent_recv_duration_seconds` | Histogram | Duration of each TCP receive |
| `smoltorrent_avg_buffer_size_kb` | Gauge | Average message buffer size |
| `smoltorrent_max_buffer_size_kb` | Gauge | Max message buffer size seen |
| `smoltorrent_store_operations_total` | Counter | Completed store operations |
| `smoltorrent_gather_operations_total` | Counter | Completed gather operations |
| `smoltorrent_transfer_errors_total{rank}` | Counter | Permanent failures by worker rank |

---

## Alerts (Gmail)

Three alert rules are provisioned automatically and fire to `rajceo2031@gmail.com`.

| Alert | Condition | Meaning |
|---|---|---|
| **Worker Transfer Errors** | any rank accumulates errors in a 5-min window | Pi down, port dead, or max retries hit |
| **Transfer p95 Too Slow** | send p95 > 2 minutes for 3 consecutive minutes | Slow SD card, network congestion, Pi under memory pressure |
| **SmolTorrent API Unreachable** | Prometheus can't scrape `:8000` for 2 minutes | FastAPI crashed or Mac went to sleep mid-run |

Alerts fire after their `for` window (2–3 min) to suppress single-scrape blips. Re-alerts are suppressed for 4 hours once acknowledged.

### One-time Gmail setup

1. **Create a Gmail app password** (required — Grafana cannot use your real Gmail password):
   Google Account → Security → 2-Step Verification → App passwords → create one named "Grafana".
   You get a 16-character password.

2. **Add it to `monitoring/docker-compose.yml`** — find this line and replace the placeholder:
   ```
   - GF_SMTP_PASSWORD=REPLACE_WITH_APP_PASSWORD
   ```

3. **Restart Grafana** to pick it up:
   ```bash
   docker restart smoltorrent-grafana
   ```

4. **Test delivery** — Grafana UI → Alerting → Contact points → gmail → **Test**.

Alert rules and contact points are provisioned from `monitoring/grafana/provisioning/alerting/` and load automatically on every Grafana start — no UI clicks needed beyond the app password.

---

## Log sources

| Source | Path on disk | How it reaches Loki |
|---|---|---|
| Mac API | `~/smoltorrent/logging/cluster-logs/syncps_api__localhost.log` | Promtail Docker container (always running) |
| Mac watcher | `~/smoltorrent/logging/cluster-logs/syncps_watcher*.log` | Promtail Docker container (always running) |
| Pi workers | `/tmp/smolcluster-logs/syncps-worker-rank{N}-{host}.log` | Promtail systemd service on each Pi (install once with `--install-pi-promtail`) |
