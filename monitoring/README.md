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

Three dashboards inside the **SmolTorrent** folder:

| Dashboard | URL | What it shows |
|---|---|---|
| **Server** | `/d/smoltorrent/server` | Transfer metrics + server system stats + cluster logs |
| **Workers** | `/d/smoltorrent-workers/workers` | Per-Pi system stats + Pi smoltorrent metrics |
| **API** | `/d/smoltorrent-api/api` | API status, ops, bandwidth, latency, errors, process stats |

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

## What's in the dashboards

### Server dashboard

| Panel | What to look for |
|---|---|
| **Bandwidth (Mbps)** | Curve that starts fast and decelerates = O(n²) memory copying bug |
| **Transfer Duration Percentiles** | p50/p90/p95/p99 send+recv — widening gap between p50 and p99 = occasional very slow transfers |
| **Transfer Errors by Worker** | A rank that keeps erroring = dead port or Pi rebooted without systemd |
| **Operations** | Total store + gather completions since API started |
| **Avg Latency (ms)** | Rolling average send/recv latency per message |
| **Buffer Size (KB)** | Sudden spike = unusually large checkpoint being transferred |
| **Bytes Transferred** | Cumulative bytes sent and received over TCP |
| **CPU Usage %** | Sustained high CPU during transfers or API calls |
| **Memory Usage** | Active + wired memory vs total |
| **Disk Free — /** | Available space on the server SSD |
| **Disk I/O MB/s** | Read/write throughput on disk0 |
| **System Load** | 1/5/15 min load averages |
| **Swap Used** | Non-zero = memory pressure |
| **Cluster Logs** | Unified log stream from all nodes — filter with `{job="smoltorrent", node="pi4-2"}` etc. |

### Workers dashboard

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
| **Store Duration Percentiles per Pi** | p50/p90/p95/p99 — rising tail = SD card degrading |
| **Send Duration Percentiles per Pi** | p50/p90/p95/p99 — rising tail = network congestion or Pi overloaded |
| **Store / Send Operations per Pi** | Op counts since worker started |
| **Store Errors per Pi** | Checksum mismatches or disk write failures |

### API dashboard

| Panel | What to look for |
|---|---|
| **API Status** | Green UP / Red DOWN |
| **Store / Gather Operations** | Total ops since API started |
| **Transfer Errors** | Total permanent failures across all ranks |
| **Process Memory** | RSS of the FastAPI process |
| **Bandwidth (Mbps)** | Send/recv bandwidth from API perspective |
| **Transfer Duration Percentiles** | p50/p90/p95/p99 send+recv |
| **Avg Latency (ms)** | Rolling average per-message latency |
| **Transfer Errors by Worker** | Per-rank error rate |
| **Bytes Transferred** | Cumulative sent + received |
| **Buffer Size (KB)** | Avg and max TCP message buffer |
| **API CPU %** | CPU consumed by the FastAPI process |
| **Open File Descriptors** | Rising without bound = FD leak |

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

Exposed by `node_exporter` on all 5 nodes (Server + 4 Pis). Powers the CPU, disk, memory, temperature, and boot-time panels in all dashboards.

`launch.sh` installs and registers `node_exporter` automatically on first run. If you need to set it up manually:

**Server (macOS — run once in your terminal):**

```bash
brew install node_exporter

# Register as a system LaunchDaemon (survives reboots — brew services is broken on macOS 26 Tahoe)
sudo tee /Library/LaunchDaemons/com.node-exporter.plist > /dev/null <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.node-exporter</string>
    <key>ProgramArguments</key>
    <array><string>/opt/homebrew/bin/node_exporter</string></array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/node-exporter.log</string>
    <key>StandardErrorPath</key><string>/tmp/node-exporter.log</string>
</dict>
</plist>
EOF
sudo chmod 644 /Library/LaunchDaemons/com.node-exporter.plist
sudo launchctl bootout system/com.node-exporter 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/com.node-exporter.plist
sudo launchctl enable system/com.node-exporter
```

Verify: `curl http://localhost:9100/metrics | grep node_boot_time_seconds`

**Pis (Linux/Raspberry Pi OS — handled automatically by `launch.sh`):**

```bash
sudo apt update && sudo apt install -y prometheus-node-exporter
sudo systemctl enable --now prometheus-node-exporter
```

All 4 Pis use systemd — the service starts at boot automatically. Verify on a Pi:

```bash
systemctl is-active node_exporter   # → active
```

---

## Alerts (Gmail)

12 alert rules are provisioned automatically and fire to the address set in `GRAFANA_ALERT_EMAIL` in `monitoring/.env`.

| Alert | Fires when | For |
|---|---|---|
| **API Down** | Prometheus can't scrape `:8000` | 2 min |
| **Pi Worker Down** | any Pi node_exporter unreachable (Pi machine down) | 2 min |
| **Server Node Exporter Down** | server node_exporter process crashed (system stats missing) | 2 min |
| **Worker Transfer Errors** | any rank accumulates transfer errors | 2 min |
| **Transfer p95 Too Slow** | send p95 > 2 min | 3 min |
| **Transfer Latency High** | avg send latency > 5 s | 3 min |
| **Transfer Bandwidth Low** | bandwidth > 0 but < 10 Mbps (active transfer running slow) | 2 min |
| **Server CPU High** | server CPU > 90% | 5 min |
| **Server Disk Low** | server root < 5 GB free | 5 min |
| **Pi CPU High** | any Pi > 90% CPU | 5 min |
| **Pi Temperature High** | any Pi SoC temp > 80°C | 2 min |
| **Pi Disk Low** | any Pi root < 2 GB free | 5 min |

Alerts fire after their `for` window to suppress single-scrape blips. Re-alerts are suppressed for 4 hours once acknowledged.

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
   GRAFANA_ALERT_EMAIL=you@gmail.com
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
