# SmolTorrent Monitoring

Prometheus + Grafana + Loki in Docker on the Server. All logs from master (API, watcher) and all 4 Pi workers stream into one Grafana view.

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
| `bash scripts/launch_monitoring.sh --daemons` | Register LaunchDaemon — stack auto-starts on every Server boot |
| `bash scripts/launch_monitoring.sh --install-pi-promtail` | Install Promtail on all 4 Pi workers via SSH |
| `bash scripts/launch_monitoring.sh --install-pi-promtail --workers 1,3` | Specific ranks only |

---

## Preflight checks

The script checks all of the following before doing anything. Failures print clearly and abort.

| Check | Auto-fix |
|---|---|
| macOS (monitoring runs on the Server master only) | — |
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
| `{job="smoltorrent", node="master"}` | Server API + watcher only |
| `{job="smoltorrent", level="ERROR"}` | Errors across all nodes |
| `{job="smoltorrent"} \|= "rank 3"` | Lines mentioning rank 3 |

The **Cluster Logs** panel on the dashboard already runs `{job="smoltorrent"}`.

---

## What's in the dashboard

### smoltorrent transfer metrics (top section)

| Panel | What to look for |
|---|---|
| **Bandwidth (Mbps)** | Curve that starts fast and decelerates = O(n²) memory copying bug |
| **Transfer Duration p95** | Rising p95 = slow worker, flaky network, or SD card degrading |
| **Transfer Errors by Worker** | A rank that keeps erroring = dead port or Pi rebooted without systemd |
| **Operations** | Total store + gather completions since API started |
| **Avg Latency (ms)** | Rolling average send/recv latency per message |
| **Buffer Size (KB)** | Sudden spike = unusually large checkpoint being transferred |
| **Bytes Transferred** | Cumulative bytes sent and received over TCP |

### Server — System Stats

| Panel | What to look for |
|---|---|
| **CPU Usage %** | Sustained high CPU during transfers or API calls |
| **Memory Usage** | Active + wired vs 16 GB total |
| **Disk Free — /** | Available space on the Server SSD |
| **Disk I/O MB/s** | Read/write throughput on disk0 |
| **System Load** | 1/5/15 min load averages |
| **Swap / Compressed** | Non-zero = memory pressure |

### Pi Workers — System Stats

| Panel | What to look for |
|---|---|
| **CPU % per Pi** | Any Pi above 80% during a transfer batch |
| **Memory per Pi** | Pi 4 has 4–8 GB — watch for pressure |
| **Disk Free per Pi** | SD card running low |
| **System Load per Pi** | Above 4 = saturation (Pi 4 is 4-core) |
| **SD Card I/O per Pi** | High write MB/s during checkpoint save |
| **CPU Temperature per Pi** | Yellow >70°C, red >80°C = throttling |
| **Bytes Received per Pi** | Cumulative shard bytes written to SD card |
| **Bytes Sent per Pi** | Cumulative shard bytes served back on gather |
| **Store Duration p95 per Pi** | Rising = SD card degrading or Pi under load |
| **Send Duration p95 per Pi** | Rising = network congestion or Pi overloaded |
| **Store / Send Operations per Pi** | Op counts since worker started |
| **Store Errors per Pi** | Checksum mismatches or disk write failures |

### API Server Stats

| Panel | What to look for |
|---|---|
| **API Status** | Green UP / Red DOWN |
| **API Process Memory** | RSS of the FastAPI process |
| **API CPU %** | CPU consumed by the API process |
| **Open File Descriptors** | Rising without bound = FD leak |

### Cluster Logs

Unified log stream from all nodes. Filter with `{job="smoltorrent", node="pi4-2"}` etc.

---

## Metrics reference

### Master API (`http://localhost:8000/metrics/`)

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

### Pi workers (`http://<pi-ip>:920{rank}/metrics`)

Exposed by `algorithms/SyncPS/worker.py` via `prometheus_client`. Each Pi exposes on port `9200 + rank` (9201–9204).

| Metric | Type | Description |
|---|---|---|
| `worker_bytes_recv_total{rank}` | Counter | Bytes received and written to disk (store_shard) |
| `worker_bytes_sent_total{rank}` | Counter | Bytes sent back to master (send_shard / gather) |
| `worker_store_ops_total{rank}` | Counter | Completed store_shard operations |
| `worker_send_ops_total{rank}` | Counter | Completed send_shard operations |
| `worker_store_errors_total{rank}` | Counter | Failed stores (checksum mismatch or disk error) |
| `worker_store_duration_seconds{rank}` | Histogram | Time to write shard to SD card |
| `worker_send_duration_seconds{rank}` | Histogram | Time to serve shard back to master |

### System metrics (`http://<node>:9100/metrics`)

Exposed by `node_exporter` on all 5 nodes (Server + 4 Pis). Installed as a systemd service on each Pi; started at boot via `smoltorrent_startup.sh` on the Server.

---

## Alerts (Gmail)

Three alert rules are provisioned automatically and fire to `rajceo2031@gmail.com`.

| Alert | Condition | Meaning |
|---|---|---|
| **Worker Transfer Errors** | any rank accumulates errors in a 5-min window | Pi down, port dead, or max retries hit |
| **Transfer p95 Too Slow** | send p95 > 2 minutes for 3 consecutive minutes | Slow SD card, network congestion, Pi under memory pressure |
| **SmolTorrent API Unreachable** | Prometheus can't scrape `:8000` for 2 minutes | FastAPI crashed or Server went to sleep mid-run |

Alerts fire after their `for` window (2–3 min) to suppress single-scrape blips. Re-alerts are suppressed for 4 hours once acknowledged.

### One-time Gmail setup

1. **Create a Gmail app password** (required — Grafana cannot use your real Gmail password):
   Google Account → Security → 2-Step Verification → App passwords → create one named "Grafana".
   You get a 16-character password.

2. **Create `monitoring/.env`** from the example file:
   ```bash
   cp monitoring/.env.example monitoring/.env
   ```
   Then edit `monitoring/.env` and fill in your values:
   ```
   GRAFANA_SMTP_USER=you@gmail.com
   GRAFANA_SMTP_PASSWORD=xxxx xxxx xxxx xxxx
   GRAFANA_SMTP_FROM=you@gmail.com
   ```
   `monitoring/.env` is gitignored — never committed.

3. **Restart Grafana** to pick it up:
   ```bash
   docker restart smoltorrent-grafana
   ```

4. **Test delivery** — Grafana UI → Alerting → Contact points → gmail → **Test**.

Alert rules and contact points are provisioned from `monitoring/grafana/provisioning/alerting/` and load automatically on every Grafana start — no UI clicks needed beyond the `.env` file.

---

## Log sources

| Source | Path on disk | How it reaches Loki |
|---|---|---|
| Server API | `~/smoltorrent/logging/cluster-logs/syncps_api__localhost.log` | Promtail Docker container (always running) |
| Server watcher | `~/smoltorrent/logging/cluster-logs/syncps_watcher*.log` | Promtail Docker container (always running) |
| Pi workers | `/tmp/smolcluster-logs/syncps-worker-rank{N}-{host}.log` | Promtail systemd service on each Pi (install once with `--install-pi-promtail`) |
