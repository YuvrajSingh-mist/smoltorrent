#!/usr/bin/env bash
# Register auto-start services for the cluster.
#
#   Pi workers (systemd):
#     bash scripts/install_worker_service.sh            # all workers
#     bash scripts/install_worker_service.sh --workers 1,3
#     bash scripts/install_worker_service.sh --uninstall
#
#   macOS monitoring stack (LaunchDaemon — run once on coordinator):
#     bash scripts/install_worker_service.sh --monitoring-daemon
#
# Requirements: SSH key at ~/.ssh/smolcluster_key, yq in PATH on master.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/configs/dev-config.yaml"
MONITORING_DIR="$PROJECT_DIR/monitoring"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/Desktop/smoltorrent}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/smolcluster_key}"
WORKER_RANKS=""
UNINSTALL=false
MONITORING_DAEMON=false
SERVICE_NAME="smoltorrent-worker"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    C_RESET="\033[0m"; C_BOLD="\033[1m"
    C_RED="\033[31m"; C_GREEN="\033[32m"; C_YELLOW="\033[33m"; C_BLUE="\033[34m"
else
    C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""
fi

info() { echo -e "${C_BLUE}${1}${C_RESET}"; }
ok()   { echo -e "${C_GREEN}${1}${C_RESET}"; }
warn() { echo -e "${C_YELLOW}${1}${C_RESET}"; }
err()  { echo -e "${C_RED}${1}${C_RESET}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)
            shift
            [[ $# -eq 0 ]] && { err "--workers requires a comma-separated rank list"; exit 1; }
            WORKER_RANKS="$1"; shift ;;
        --uninstall)
            UNINSTALL=true; shift ;;
        --monitoring-daemon)
            MONITORING_DAEMON=true; shift ;;
        --ssh-key)
            shift
            [[ $# -eq 0 ]] && { err "--ssh-key requires a path"; exit 1; }
            SSH_KEY="$1"; shift ;;
        *)
            err "Unknown option: $1"
            warn "Usage: $0 [--workers <rank,...>] [--uninstall] [--monitoring-daemon] [--ssh-key <path>]"
            exit 1 ;;
    esac
done

# ── --monitoring-daemon: register monitoring stack as macOS LaunchDaemon ──────

if [[ "$MONITORING_DAEMON" == "true" ]]; then
    PLIST_LABEL="com.smoltorrent.monitoring"
    PLIST_DST="/Library/LaunchDaemons/${PLIST_LABEL}.plist"
    STARTUP_SCRIPT_DST="/usr/local/bin/smoltorrent_monitoring_startup.sh"
    CURRENT_USER="$(whoami)"

    info "Writing startup script to $STARTUP_SCRIPT_DST..."
    sudo tee "$STARTUP_SCRIPT_DST" > /dev/null <<STARTUP
#!/bin/bash
# Runs at boot via LaunchDaemon: waits for colima, then brings up monitoring stack.
set -euo pipefail
LOG=/tmp/smoltorrent-monitoring-startup.log
log() { echo "[\$(date)] smoltorrent-monitoring: \$*" | tee -a "\$LOG"; }
log "startup triggered"
sleep 10
PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH"
for i in \$(seq 1 60); do
    if colima status 2>&1 | grep -q "colima is running"; then
        log "colima ready"; break
    fi
    [[ \$i -eq 1 ]] && { log "starting colima..."; colima start &>/dev/null &; }
    sleep 5
    [[ \$i -eq 60 ]] && { log "ERROR: colima did not start after 5 min"; exit 1; }
done
log "starting docker-compose..."
cd "$MONITORING_DIR"
docker-compose up -d >> "\$LOG" 2>&1
log "monitoring stack launched"
STARTUP
    sudo chmod +x "$STARTUP_SCRIPT_DST"
    ok "Startup script written"

    info "Writing LaunchDaemon plist to $PLIST_DST..."
    sudo tee "$PLIST_DST" > /dev/null <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>${PLIST_LABEL}</string>
    <key>UserName</key><string>${CURRENT_USER}</string>
    <key>ProgramArguments</key>
    <array><string>/bin/bash</string><string>${STARTUP_SCRIPT_DST}</string></array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><false/>
    <key>StandardOutPath</key><string>/tmp/smoltorrent-monitoring-startup.log</string>
    <key>StandardErrorPath</key><string>/tmp/smoltorrent-monitoring-startup.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HOME</key><string>/Users/${CURRENT_USER}</string>
    </dict>
</dict>
</plist>
PLIST
    sudo chmod 644 "$PLIST_DST"
    ok "Plist written"

    info "Registering with launchctl..."
    sudo launchctl bootout system/"$PLIST_LABEL" 2>/dev/null || true
    sudo launchctl bootstrap system "$PLIST_DST"
    sudo launchctl enable system/"$PLIST_LABEL"
    ok "LaunchDaemon registered — monitoring stack will auto-start on boot"

    echo
    echo -e "  Verify:    ${C_YELLOW}sudo launchctl print system/${PLIST_LABEL}${C_RESET}"
    echo -e "  Boot log:  ${C_YELLOW}tail -f /tmp/smoltorrent-monitoring-startup.log${C_RESET}"
    echo -e "  Uninstall: ${C_YELLOW}sudo launchctl bootout system/${PLIST_LABEL} && sudo rm $PLIST_DST $STARTUP_SCRIPT_DST${C_RESET}"
    exit 0
fi

# ── Pi worker systemd service ──────────────────────────────────────────────────

if ! command -v yq >/dev/null 2>&1; then
    err "yq is required to parse $CONFIG_FILE (brew install yq)"
    exit 1
fi

[[ ! -f "$CONFIG_FILE" ]] && { err "Config not found: $CONFIG_FILE"; exit 1; }

rank_selected() {
    local rank="$1"
    [[ -z "$WORKER_RANKS" ]] && return 0
    local IFS=','
    for r in $WORKER_RANKS; do [[ "$r" == "$rank" ]] && return 0; done
    return 1
}

ssh_run() {
    local host="$1"; shift
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "$host" "$@"
}

# Read workers from config
WORKER_ENTRIES=()
while IFS= read -r entry; do
    [[ -n "$entry" && "$entry" != "null" ]] && WORKER_ENTRIES+=("$entry")
done < <(yq '.devices_config.workers[] | (.host) + ":" + (.rank | tostring) + ":" + (.port | tostring)' "$CONFIG_FILE")

[[ ${#WORKER_ENTRIES[@]} -eq 0 ]] && { err "No workers found in $CONFIG_FILE"; exit 1; }

info "${C_BOLD}SSH key :${C_RESET} $SSH_KEY"
info "${C_BOLD}Remote  :${C_RESET} $REMOTE_PROJECT_DIR"
echo ""

for entry in "${WORKER_ENTRIES[@]}"; do
    host="${entry%%:*}"; rest="${entry#*:}"
    rank="${rest%%:*}"; port="${rest##*:}"

    rank_selected "$rank" || { warn "  rank $rank ($host) — skipped"; continue; }

    if [[ "$UNINSTALL" == "true" ]]; then
        info "Uninstalling $SERVICE_NAME@${rank}.service on $host (rank $rank)..."
        ssh_run "$host" bash -s <<EOF
set -euo pipefail
SERVICE="${SERVICE_NAME}@${rank}.service"
sudo systemctl stop  "\$SERVICE" 2>/dev/null || true
sudo systemctl disable "\$SERVICE" 2>/dev/null || true
sudo rm -f "/etc/systemd/system/${SERVICE_NAME}@.service"
sudo systemctl daemon-reload
echo "Removed \$SERVICE"
EOF
        ok "  rank $rank ($host) — service removed"
        continue
    fi

    info "Installing $SERVICE_NAME@${rank}.service on $host (rank $rank, port $port)..."

    # Detect where uv-installed Python lives on the Pi
    # Installs the unit file using a template (@) instance so one unit file
    # covers all ranks; each instance is parameterised by rank (%i).
    ssh_run "$host" REMOTE_PROJECT_DIR="$REMOTE_PROJECT_DIR" RANK="$rank" PORT="$port" SERVICE_NAME="$SERVICE_NAME" bash -s <<'REMOTE'
set -euo pipefail

resolved_dir=$(eval echo "$REMOTE_PROJECT_DIR")
python_bin="$resolved_dir/.venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: venv python not found at $python_bin — run launch.sh first to sync and create the venv"
    exit 1
fi

# Write the template unit file (one file, many instances via rank)
sudo tee /etc/systemd/system/${SERVICE_NAME}@.service > /dev/null <<UNIT
[Unit]
Description=SmolTorrent worker (rank %i)
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${resolved_dir}
ExecStart=${python_bin} algorithms/SyncPS/worker.py %i $(hostname)
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}@${RANK}.service
sudo systemctl restart ${SERVICE_NAME}@${RANK}.service

# Give it a moment then check status
sleep 2
if systemctl is-active --quiet ${SERVICE_NAME}@${RANK}.service; then
    echo "OK: ${SERVICE_NAME}@${RANK} is running"
else
    echo "WARN: service enabled but not yet active — check: journalctl -u ${SERVICE_NAME}@${RANK} -n 30"
fi
REMOTE

    ok "  rank $rank ($host) — service installed and started"
done

echo ""
if [[ "$UNINSTALL" == "true" ]]; then
    ok "Done. Workers will no longer auto-start on reboot."
else
    ok "Done. Workers will now auto-start on reboot and restart on crash."
    info "Check status : ssh -i $SSH_KEY pi4-1 'systemctl status ${SERVICE_NAME}@1'"
    info "Live logs    : ssh -i $SSH_KEY pi4-1 'journalctl -u ${SERVICE_NAME}@1 -f'"
    info "Restart      : ssh -i $SSH_KEY pi4-1 'sudo systemctl restart ${SERVICE_NAME}@1'"
    info "Uninstall    : bash scripts/install_worker_service.sh --uninstall"
fi
