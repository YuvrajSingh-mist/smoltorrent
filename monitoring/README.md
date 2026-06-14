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

Pi workers write logs to `~/Desktop/smoltorrent/logging/cluster-logs/` on each Pi. Install Promtail once to ship them to Loki:

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

### Boot exporter (`http://<node>:9101/metrics`)

Runs on all 5 nodes. Source: `utils/boot_exporter.py`.

| Metric | Type | Description |
|---|---|---|
| `smoltorrent_boot_time_ms` | Gauge | Unix timestamp of last OS boot in milliseconds |

---

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

Exposed by `node_exporter` on all 5 nodes (Server + 4 Pis). Powers the CPU, disk, memory, temperature panels in all dashboards.

`bash scripts/launch.sh --daemons` installs and registers `node_exporter` automatically. If you need to do it manually:

**Server (macOS):**

```bash
brew install node_exporter
sudo chmod 644 /Library/LaunchDaemons/com.node-exporter.plist
sudo launchctl bootout system/com.node-exporter 2>/dev/null || true
sudo launchctl bootstrap system /Library/LaunchDaemons/com.node-exporter.plist
sudo launchctl enable system/com.node-exporter
```

**Pis (Linux — handled automatically by `launch.sh`):**

```bash
sudo apt update && sudo apt install -y prometheus-node-exporter
sudo systemctl enable --now prometheus-node-exporter
```

Verify: `curl http://<node>:9100/metrics | grep node_boot`

---

### Boot time metric (`http://<node>:9101/metrics`)

`utils/boot_exporter.py` — a tiny cross-platform Prometheus exporter that reads the OS boot timestamp directly (`sysctl kern.boottime` on macOS, `/proc/stat` on Linux) and exposes it as `smoltorrent_boot_time_ms`. This powers the **Server Last Boot** and **API Process Last Start** panels.

It runs on port 9101 on all 5 nodes and is registered to survive reboots automatically.

**Server (macOS) — registered by `launch.sh --daemons`:**

```bash
sudo bash scripts/launch.sh --daemons
```

Registers `com.smoltorrent.boot-exporter` as a system LaunchDaemon. Verify:

```bash
curl http://localhost:9101/metrics | grep smoltorrent_boot_time_ms
tail -f /tmp/smoltorrent-boot-exporter.log
```

**Pis (Linux) — deployed by `launch.sh` via SSH:**

`launch.sh` SSHes into each Pi, writes `/etc/systemd/system/smoltorrent-boot-exporter.service`, and enables it. To install manually on a Pi:

```bash
# On the Pi
cat <<EOF | sudo tee /etc/systemd/system/smoltorrent-boot-exporter.service
[Unit]
Description=smoltorrent boot time exporter (port 9101)
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$HOME/Desktop/smoltorrent
ExecStart=$HOME/.local/bin/uv run $HOME/Desktop/smoltorrent/utils/boot_exporter.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now smoltorrent-boot-exporter
```

Verify on any Pi:

```bash
systemctl is-active smoltorrent-boot-exporter   # → active
curl http://localhost:9101/metrics | grep smoltorrent_boot_time_ms
```

---

## Alerts (Telegram)

12 alert rules are provisioned automatically and fire to your Telegram account via a bot.

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

### One-time Telegram setup

1. **Create a Telegram bot** — message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`, give it a name (e.g. `SmolTorrent Alerts`). Copy the bot token.

2. **Get your chat ID** — message [@userinfobot](https://t.me/userinfobot) on Telegram. Copy the numeric chat ID.

3. **Create `monitoring/.env`** from the example file:
   ```bash
   cp monitoring/.env.example monitoring/.env
   ```
   Then edit `monitoring/.env` and fill in your values:
   ```
   GRAFANA_TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   GRAFANA_TELEGRAM_CHAT_ID=123456789
   ```
   `monitoring/.env` is gitignored — never committed.

4. **Restart the stack** to pick up the new config:
   ```bash
   cd monitoring && docker compose down && docker compose up -d
   ```

5. **Test delivery** — Grafana UI → Alerting → Contact points → telegram → **Test**. You should get a Telegram message instantly.

Alert rules and contact points are provisioned from `monitoring/grafana/provisioning/alerting/` and load automatically on every Grafana start — no UI clicks needed beyond the `.env` file.

---

## Log sources

| Source | Path on disk | How it reaches Loki |
|---|---|---|
| Server API | `~/smoltorrent/logging/cluster-logs/syncps_api__localhost.log` | Promtail Docker container (always running) |
| Server watcher | `~/smoltorrent/logging/cluster-logs/syncps_watcher*.log` | Promtail Docker container (always running) |
| Pi workers | `~/Desktop/smoltorrent/logging/cluster-logs/syncps-worker-rank{N}-{host}.log` | Promtail systemd service on each Pi (install once with `--install-pi-promtail`) |
