#!/bin/bash
# Launch the SmolTorrent monitoring stack (Prometheus + Grafana + Loki + Promtail).
#
# Usage:
#   bash scripts/launch_monitoring.sh                            # start stack
#   bash scripts/launch_monitoring.sh --down                     # stop stack
#   bash scripts/launch_monitoring.sh --daemons                  # register auto-start at boot (macOS)
#   bash scripts/launch_monitoring.sh --install-pi-promtail      # install promtail on all Pi workers
#   bash scripts/launch_monitoring.sh --install-pi-promtail --workers 1,3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MONITORING_DIR="$PROJECT_DIR/monitoring"
CONFIG_FILE="$PROJECT_DIR/configs/config.yaml"
LOG_DIR="$HOME/smoltorrent/logging/cluster-logs"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/smolcluster_key}"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    C_RESET="\033[0m"; C_GREEN="\033[32m"; C_RED="\033[31m"
    C_YELLOW="\033[33m"; C_BLUE="\033[34m"; C_BOLD="\033[1m"
else
    C_RESET=""; C_GREEN=""; C_RED=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""
fi

ok()   { echo -e "  ${C_GREEN}✓${C_RESET} $1"; }
fail() { echo -e "  ${C_RED}✗${C_RESET} $1"; }
info() { echo -e "${C_BLUE}→${C_RESET} $1"; }
warn() { echo -e "  ${C_YELLOW}!${C_RESET} $1"; }
die()  { echo -e "${C_RED}ERROR:${C_RESET} $1" >&2; exit 1; }

# ── arg parsing ───────────────────────────────────────────────────────────────

MODE="up"
WORKER_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --down)                 MODE="down" ;;
        --daemons)              MODE="daemons" ;;
        --install-pi-promtail)  MODE="pi" ;;
        --workers)              shift; WORKER_FILTER="$1" ;;
        *) die "Unknown arg: $1. Valid: --down | --daemons | --install-pi-promtail [--workers N,M]" ;;
    esac
    shift
done

# ── preflight checks (run for all modes that need the env) ───────────────────

preflight() {
    local ok=true
    info "Preflight checks..."

    # macOS only
    if [[ "$(uname)" != "Darwin" ]]; then
        fail "This script is macOS-only (monitoring stack runs on the Mac master)"
        ok=false
    fi

    # colima installed
    if ! command -v colima &>/dev/null; then
        fail "colima not found — install: brew install colima"
        ok=false
    else
        ok "colima installed"
    fi

    # docker installed
    if ! command -v docker &>/dev/null; then
        fail "docker not found — install: brew install docker"
        ok=false
    else
        ok "docker installed"
    fi

    # docker-compose installed (auto-install if missing)
    if ! command -v docker-compose &>/dev/null; then
        warn "docker-compose not found — installing via brew..."
        brew install docker-compose &>/dev/null && ok "docker-compose installed" || { fail "docker-compose install failed"; ok=false; }
    else
        ok "docker-compose installed"
    fi

    # python3 + yaml (needed to parse config)
    if ! python3 -c "import yaml" &>/dev/null; then
        fail "python3-yaml not found — install: pip3 install pyyaml"
        ok=false
    else
        ok "python3 + yaml"
    fi

    # config.yaml
    if [[ ! -f "$CONFIG_FILE" ]]; then
        fail "configs/config.yaml not found at $CONFIG_FILE"
        ok=false
    else
        ok "config.yaml found"
    fi

    # monitoring dir and docker-compose.yml
    if [[ ! -f "$MONITORING_DIR/docker-compose.yml" ]]; then
        fail "monitoring/docker-compose.yml not found at $MONITORING_DIR"
        ok=false
    else
        ok "monitoring/docker-compose.yml found"
    fi

    # required monitoring sub-configs
    for f in prometheus/prometheus.yml loki/loki-config.yaml promtail/promtail-mac.yaml; do
        if [[ ! -f "$MONITORING_DIR/$f" ]]; then
            fail "monitoring/$f missing"
            ok=false
        fi
    done

    # log dir writable (create if absent)
    mkdir -p "$LOG_DIR" 2>/dev/null || { fail "Cannot create log dir $LOG_DIR"; ok=false; }
    if [[ ! -w "$LOG_DIR" ]]; then
        fail "Log dir not writable: $LOG_DIR"
        ok=false
    else
        ok "log dir writable ($LOG_DIR)"
    fi

    [[ "$ok" == "true" ]] || die "Fix the issues above and re-run."
}

preflight_pi() {
    # Extra checks needed only for --install-pi-promtail
    local ok=true
    if [[ ! -f "$SSH_KEY" ]]; then
        fail "SSH key not found: $SSH_KEY  (override with SSH_KEY=... bash $0 --install-pi-promtail)"
        ok=false
    else
        ok "SSH key: $SSH_KEY"
    fi
    if [[ ! -f "$MONITORING_DIR/promtail/promtail-pi.yaml" ]]; then
        fail "monitoring/promtail/promtail-pi.yaml missing"
        ok=false
    else
        ok "promtail-pi.yaml found"
    fi
    [[ "$ok" == "true" ]] || die "Fix the issues above and re-run."
}

# ── helpers ───────────────────────────────────────────────────────────────────

read_config_workers() {
    python3 - "$CONFIG_FILE" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for w in cfg["devices_config"]["workers"]:
    print(w["rank"], w.get("host", w.get("device")), w["ip"], w["port"])
PY
}

read_master_ip() {
    python3 - "$CONFIG_FILE" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["devices_config"]["master"][0]["ip"])
PY
}

start_colima() {
    if colima status 2>&1 | grep -q "colima is running"; then
        ok "colima already running"
        return
    fi
    info "Starting colima (takes ~60s)..."
    colima start 2>&1 | grep -E "level=(info|fatal)" | sed 's/.*msg="\(.*\)".*/  \1/' &
    local pid=$!
    for i in $(seq 1 30); do
        sleep 3
        if colima status 2>&1 | grep -q "colima is running"; then
            wait $pid 2>/dev/null || true
            ok "colima started"
            return
        fi
    done
    wait $pid 2>/dev/null || true
    die "colima failed to start after 90s — run 'colima start' manually and retry"
}

wait_for() {
    local name=$1 url=$2 keyword=$3 timeout=${4:-40}
    for i in $(seq 1 $timeout); do
        if curl -sf "$url" 2>/dev/null | grep -q "$keyword"; then
            ok "$name"
            return 0
        fi
        sleep 1
    done
    fail "$name (not ready after ${timeout}s — check: cd monitoring && docker-compose logs $name)"
    return 1
}

# ── --down ────────────────────────────────────────────────────────────────────

if [[ "$MODE" == "down" ]]; then
    info "Stopping monitoring stack..."
    cd "$MONITORING_DIR" && docker-compose down
    ok "Stack stopped (volumes preserved — data survives restart)"
    exit 0
fi

# ── --daemons: register auto-start LaunchDaemon on macOS ─────────────────────

if [[ "$MODE" == "daemons" ]]; then
    preflight

    PLIST_LABEL="com.smoltorrent.monitoring"
    PLIST_DST="/Library/LaunchDaemons/${PLIST_LABEL}.plist"
    STARTUP_SCRIPT_DST="/usr/local/bin/smoltorrent_monitoring_startup.sh"
    CURRENT_USER="$(whoami)"

    # Write the startup script that launchd will run at boot
    info "Writing startup script to $STARTUP_SCRIPT_DST..."
    sudo tee "$STARTUP_SCRIPT_DST" > /dev/null <<STARTUP
#!/bin/bash
# Auto-generated by launch_monitoring.sh --daemons
# Runs at boot via LaunchDaemon: waits for colima, then brings up monitoring stack.
set -euo pipefail
LOG=/tmp/smoltorrent-monitoring-startup.log
log() { echo "[\$(date)] smoltorrent-monitoring: \$*" | tee -a "\$LOG"; }

log "startup triggered"

# Give the system a moment to settle after boot
sleep 10

# Start colima (Docker VM) — retry up to 5 min
PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH"
for i in \$(seq 1 60); do
    if colima status 2>&1 | grep -q "colima is running"; then
        log "colima ready"
        break
    fi
    if [[ \$i -eq 1 ]]; then
        log "starting colima..."
        colima start &>/dev/null &
    fi
    sleep 5
    if [[ \$i -eq 60 ]]; then
        log "ERROR: colima did not start after 5 min — aborting"
        exit 1
    fi
done

# Bring monitoring stack up (restart: unless-stopped handles crashes after this)
log "starting docker-compose..."
cd "$MONITORING_DIR"
docker-compose up -d >> "\$LOG" 2>&1
log "monitoring stack launched"
STARTUP
    sudo chmod +x "$STARTUP_SCRIPT_DST"
    ok "Startup script written"

    # Write plist
    # macOS 26 Tahoe: LaunchAgents silently ignored; /Library/LaunchDaemons + bootstrap system works.
    # TCC blocks system daemons from ~/Desktop etc — script lives in /usr/local/bin (TCC-safe).
    info "Writing LaunchDaemon plist to $PLIST_DST..."
    sudo tee "$PLIST_DST" > /dev/null <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>
    <key>UserName</key>
    <string>${CURRENT_USER}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${STARTUP_SCRIPT_DST}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>/tmp/smoltorrent-monitoring-startup.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/smoltorrent-monitoring-startup.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HOME</key>
        <string>/Users/${CURRENT_USER}</string>
    </dict>
</dict>
</plist>
PLIST
    sudo chmod 644 "$PLIST_DST"
    ok "Plist written"

    # Register
    info "Registering with launchctl..."
    sudo launchctl bootout system/"$PLIST_LABEL" 2>/dev/null || true
    sudo launchctl bootstrap system "$PLIST_DST"
    sudo launchctl enable system/"$PLIST_LABEL"
    ok "LaunchDaemon registered"

    echo
    echo -e "${C_BOLD}Monitoring will auto-start on every boot.${C_RESET}"
    echo -e "  Verify:    ${C_YELLOW}sudo launchctl print system/${PLIST_LABEL}${C_RESET}"
    echo -e "  Boot log:  ${C_YELLOW}tail -f /tmp/smoltorrent-monitoring-startup.log${C_RESET}"
    echo -e "  Uninstall: ${C_YELLOW}sudo launchctl bootout system/${PLIST_LABEL} && sudo rm $PLIST_DST $STARTUP_SCRIPT_DST${C_RESET}"
    echo
    echo -e "  Note: docker containers have ${C_BLUE}restart: unless-stopped${C_RESET} — crashes are handled"
    echo -e "  automatically by Docker without needing launchd to re-fire."
    exit 0
fi

# ── --install-pi-promtail ─────────────────────────────────────────────────────

if [[ "$MODE" == "pi" ]]; then
    preflight
    preflight_pi

    echo -e "${C_BOLD}Installing Promtail on Pi workers${C_RESET}"
    echo

    LOKI_IP="$(read_master_ip)"
    PROMTAIL_CFG="$MONITORING_DIR/promtail/promtail-pi.yaml"

    while IFS=" " read -r rank host ip port; do
        if [[ -n "$WORKER_FILTER" ]]; then
            IFS=',' read -ra wanted <<< "$WORKER_FILTER"
            match=0
            for w in "${wanted[@]}"; do [[ "$w" == "$rank" ]] && match=1; done
            [[ $match -eq 0 ]] && continue
        fi

        info "rank $rank ($host $ip)"

        # Test SSH reachability before proceeding
        if ! ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$host" true 2>/dev/null; then
            fail "rank $rank ($host) — SSH unreachable, skipping"
            continue
        fi

        # NODE_LABEL = SSH alias (strip @ip suffix if host field is user@ip)
        node_label="${host%%@*}"
        # base64-encode config to survive heredoc expansion (yaml regex has special chars)
        # PI_USER (linux username) is unknown here — resolved on the Pi via whoami after writing
        CFG_B64=$(sed -e "s|LOKI_IP|$LOKI_IP|g" \
                      -e "s|NODE_LABEL|$node_label|g" \
                      -e "s|RANK|$rank|g" \
                      "$PROMTAIL_CFG" | base64)

        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$host" bash -s <<REMOTE
set -e
if ! command -v promtail &>/dev/null; then
    arch=\$(uname -m)
    [[ "\$arch" == "aarch64" ]] && arch=arm64 || arch=amd64
    # Standalone promtail binaries dropped after v2.9.x
    VER="v2.9.10"
    URL="https://github.com/grafana/loki/releases/download/\${VER}/promtail-linux-\${arch}.zip"
    echo "  downloading promtail \${VER#v} (\$arch)..."
    curl -fsSL --retry 3 -o /tmp/promtail.zip "\$URL"
    unzip -oq /tmp/promtail.zip -d /tmp/
    sudo mv "/tmp/promtail-linux-\${arch}" /usr/local/bin/promtail
    sudo chmod +x /usr/local/bin/promtail
    rm -f /tmp/promtail.zip
    echo "  installed \$(promtail --version 2>&1 | head -1)"
else
    echo "  promtail already installed: \$(promtail --version 2>&1 | head -1)"
fi

sudo mkdir -p /etc/promtail
mkdir -p /tmp/smolcluster-logs

echo '$CFG_B64' | base64 -d | sudo tee /etc/promtail/config.yaml >/dev/null
# Resolve PI_USER to the actual linux username and fix the log path
sudo sed -i "s|PI_USER|$(whoami)|g" /etc/promtail/config.yaml

sudo tee /etc/systemd/system/smoltorrent-promtail.service >/dev/null <<SVC
[Unit]
Description=Promtail log shipper for SmolTorrent
After=network-online.target
[Service]
ExecStart=/usr/local/bin/promtail -config.file=/etc/promtail/config.yaml
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
SVC

sudo systemctl daemon-reload
sudo systemctl enable smoltorrent-promtail
sudo systemctl restart smoltorrent-promtail
sleep 2
sudo systemctl is-active smoltorrent-promtail
REMOTE
        ok "rank $rank ($host) — promtail active, shipping to Loki at $LOKI_IP"
    done < <(read_config_workers)

    echo
    echo -e "${C_BOLD}Done.${C_RESET} Pi logs appear in Grafana → Explore → Loki:"
    echo -e "  All nodes:  ${C_YELLOW}{job=\"smoltorrent\"}${C_RESET}"
    echo -e "  One Pi:     ${C_YELLOW}{job=\"smoltorrent\", node=\"pi4-2\"}${C_RESET}"
    echo -e "  Errors:     ${C_YELLOW}{job=\"smoltorrent\", level=\"ERROR\"}${C_RESET}"
    exit 0
fi

# ── default: bring stack up ───────────────────────────────────────────────────

echo -e "${C_BOLD}SmolTorrent Monitoring Setup${C_RESET}"
echo

preflight
start_colima

info "Starting monitoring stack..."
cd "$MONITORING_DIR"
docker-compose up -d 2>&1 | grep -E "(Starting|Started|Created|Pulling|pulled|error)" | sed 's/^/  /' || true

info "Waiting for services to be healthy..."
wait_for "Prometheus" "http://localhost:9090/-/ready"   "is Ready"  30
wait_for "Loki"       "http://localhost:3100/ready"      "ready"     45
wait_for "Grafana"    "http://localhost:3000/api/health" '"ok"'      30

if docker-compose ps promtail 2>/dev/null | grep -q "Up"; then
    ok "Promtail (tailing $LOG_DIR)"
else
    fail "Promtail — check: cd monitoring && docker-compose logs promtail"
fi

echo
echo -e "${C_BOLD}All services up.${C_RESET}"
echo
echo -e "  Grafana   → ${C_BLUE}http://127.0.0.1:3000${C_RESET}  (admin / smoltorrent)"
echo -e "  Dashboard → ${C_BLUE}http://127.0.0.1:3000/d/smoltorrent/smoltorrent${C_RESET}"
echo -e "  Logs      → Grafana → Explore → Loki → {job=\"smoltorrent\"}"
echo -e "  Metrics   → ${C_BLUE}http://127.0.0.1:9090${C_RESET}"
echo
echo -e "  Pi logs   → run once: ${C_YELLOW}bash scripts/launch_monitoring.sh --install-pi-promtail${C_RESET}"
echo -e "  Auto-boot → run once: ${C_YELLOW}bash scripts/launch_monitoring.sh --daemons${C_RESET}"
echo -e "  Stop:               ${C_YELLOW}bash scripts/launch_monitoring.sh --down${C_RESET}"
