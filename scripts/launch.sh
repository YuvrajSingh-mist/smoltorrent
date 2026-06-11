#!/bin/bash
# Start the smoltorrent cluster across all nodes in configs/config.yaml.
# Assumes bootstrap.sh has already been run on every host (deps installed, venv ready).
# Rsyncs latest code to workers, kills stale sessions, then starts everything in tmux.
#
# Usage: launch.sh [--dry-run] [--api-only] [--daemons] [--workers <rank,...>] [--ext <exts>]
#
#   (default)          rsync code -> launch syncps_api, syncps_watcher, syncps_worker_N
#   --api-only         heartbeat-check workers, then launch API only
#   --workers 1,3      launch only the specified worker ranks
#   --ext .pt,.pth     override file extensions the watcher monitors (default: .safetensors)
#   --daemons          one-time macOS setup: register smoltorrent, node_exporter, and
#                      boot_exporter as LaunchDaemons so they survive reboots
#   --dry-run          print what would run without executing anything
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/configs/config.yaml"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/Desktop/smoltorrent}"
DRY_RUN=false
API_ONLY=false
DAEMONS=false
WORKER_RANKS=""
WATCH_EXT=".safetensors"

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
        --dry-run)  DRY_RUN=true; shift ;;
        --api-only) API_ONLY=true; shift ;;
        --daemons)  DAEMONS=true; shift ;;
        --ext)
            shift
            [[ $# -eq 0 ]] && { err "--ext requires a value (e.g. --ext .safetensors,.pth)"; exit 1; }
            WATCH_EXT="$1"; shift ;;
        --workers)
            shift
            [[ $# -eq 0 ]] && { err "--workers requires a comma-separated rank list (e.g. --workers 1,3)"; exit 1; }
            WORKER_RANKS="$1"; shift ;;
        *)
            err "Unknown option: $1"
            warn "Usage: $0 [--dry-run] [--api-only] [--daemons] [--workers <rank,...>] [--ext <exts>]"
            exit 1 ;;
    esac
done

# ── --daemons: one-time macOS LaunchDaemon registration ───────────────────────

if [[ "$DAEMONS" == "true" ]]; then
    STARTUP_SCRIPT="$SCRIPT_DIR/smoltorrent_startup.sh"
    PLIST_LABEL="com.smoltorrent.startup"
    PLIST_DST="/Library/LaunchDaemons/${PLIST_LABEL}.plist"
    SCRIPT_DST="/usr/local/bin/smoltorrent_startup.sh"

    # macOS 26 Tahoe notes:
    #   - launchctl load          -> SIGABRT exit 134 (API removed)
    #   - launchctl bootstrap gui -> error 125 (GUI domain broken in beta)
    #   - ~/Library/LaunchAgents  -> silently ignored (needs SMAppService from Swift)
    #   - /Library/LaunchDaemons  + sudo launchctl enable + bootstrap system -> WORKS
    #
    # TCC blocks system daemons from ~/Desktop, ~/Documents, ~/Downloads.
    # Script lives at /usr/local/bin/ to sidestep TCC.

    info "Copying startup script to /usr/local/bin/ (TCC-safe location)..."
    sudo cp "$STARTUP_SCRIPT" "$SCRIPT_DST"
    sudo chmod +x "$SCRIPT_DST"

    CURRENT_USER="$(whoami)"

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
        <string>${SCRIPT_DST}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>/tmp/smoltorrent-startup.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/smoltorrent-startup.log</string>
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

    info "Registering with launchctl..."
    sudo launchctl bootout system/"$PLIST_LABEL" 2>/dev/null || true
    sudo launchctl bootstrap system "$PLIST_DST"
    sudo launchctl enable system/"$PLIST_LABEL"

    ok "LaunchDaemon registered - smoltorrent will auto-start on next boot."
    info "Verify:  sudo launchctl print system/${PLIST_LABEL}"
    info "Logs:    tail -f /tmp/smoltorrent-startup.log"

    NODE_EXP_PLIST="/Library/LaunchDaemons/com.node-exporter.plist"
    NODE_EXP_LABEL="com.node-exporter"

    if ! command -v node_exporter >/dev/null 2>&1; then
        if ! command -v brew >/dev/null 2>&1; then
            err "Homebrew required to install node_exporter - skipping"
        else
            info "Installing node_exporter via Homebrew..."
            brew install node_exporter
        fi
    fi

    if command -v node_exporter >/dev/null 2>&1; then
        NODE_EXP_BIN="$(command -v node_exporter)"
        info "Registering node_exporter LaunchDaemon -> $NODE_EXP_PLIST"
        sudo tee "$NODE_EXP_PLIST" > /dev/null <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${NODE_EXP_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${NODE_EXP_BIN}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/node-exporter.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/node-exporter.log</string>
</dict>
</plist>
PLIST
        sudo chmod 644 "$NODE_EXP_PLIST"
        sudo launchctl bootout system/"$NODE_EXP_LABEL" 2>/dev/null || true
        sudo launchctl bootstrap system "$NODE_EXP_PLIST"
        sudo launchctl enable system/"$NODE_EXP_LABEL"
        ok "node_exporter LaunchDaemon registered - metrics on port 9100 survive reboots."
        info "Verify:  curl http://localhost:9100/metrics | grep node_boot_time_seconds"
        info "Logs:    tail -f /tmp/node-exporter.log"
    fi

    BOOT_EXP_PLIST="/Library/LaunchDaemons/com.smoltorrent.boot-exporter.plist"
    BOOT_EXP_LABEL="com.smoltorrent.boot-exporter"
    BOOT_EXP_SCRIPT="$PROJECT_DIR/utils/boot_exporter.py"
    BOOT_EXP_UV="$(command -v uv || echo /opt/homebrew/bin/uv)"

    info "Registering boot_exporter LaunchDaemon -> $BOOT_EXP_PLIST"
    sudo tee "$BOOT_EXP_PLIST" > /dev/null <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${BOOT_EXP_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${BOOT_EXP_UV}</string>
        <string>run</string>
        <string>${BOOT_EXP_SCRIPT}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>UserName</key>
    <string>$(whoami)</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/smoltorrent-boot-exporter.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/smoltorrent-boot-exporter.log</string>
</dict>
</plist>
PLIST
    sudo chmod 644 "$BOOT_EXP_PLIST"
    sudo launchctl bootout system/"$BOOT_EXP_LABEL" 2>/dev/null || true
    sudo launchctl bootstrap system "$BOOT_EXP_PLIST"
    sudo launchctl enable system/"$BOOT_EXP_LABEL"
    ok "boot_exporter LaunchDaemon registered - boot time metric on port 9101 survives reboots."
    info "Verify:  curl http://localhost:9101/metrics | grep smoltorrent_boot_time_ms"
    info "Logs:    tail -f /tmp/smoltorrent-boot-exporter.log"

    exit 0
fi

# ── Preflight ──────────────────────────────────────────────────────────────────

if ! command -v yq >/dev/null 2>&1; then
    err "yq is required to parse $CONFIG_FILE  (brew install yq)"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    err "Config file not found: $CONFIG_FILE"
    exit 1
fi

if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
    err "No .venv found — run bootstrap first:  bash scripts/bootstrap.sh"
    exit 1
fi

MASTER_HOST="$(yq '.devices_config.master[0].host // .devices_config.master.host' "$CONFIG_FILE")"
if [[ -z "$MASTER_HOST" || "$MASTER_HOST" == "null" ]]; then
    err "Could not read master host from $CONFIG_FILE"
    exit 1
fi

WORKER_ENTRIES=()
while IFS= read -r entry; do
    [[ -n "$entry" && "$entry" != "null" ]] && WORKER_ENTRIES+=("$entry")
done < <(yq '.devices_config.workers[] | (.device // .host) + ":" + (.rank | tostring)' "$CONFIG_FILE")

if [[ ${#WORKER_ENTRIES[@]} -eq 0 ]]; then
    err "No workers found in $CONFIG_FILE"
    exit 1
fi

# ── Helpers ────────────────────────────────────────────────────────────────────

is_local_host() {
    local host="$1"
    [[ "$host" == "localhost" || "$host" == "127.0.0.1" \
       || "$host" == "$(hostname -s)" || "$host" == "$(hostname)" ]]
}

rank_selected() {
    local rank="$1"
    [[ -z "$WORKER_RANKS" ]] && return 0
    local IFS=','
    for r in $WORKER_RANKS; do [[ "$r" == "$rank" ]] && return 0; done
    return 1
}

launch_on_node() {
    local host="$1"
    local session="$2"
    local run_cmd="$3"

    if [[ "$DRY_RUN" == "true" ]]; then
        warn "[DRY RUN] $host :: $session -> $run_cmd"
        return 0
    fi

    if is_local_host "$host"; then
        mkdir -p "$PROJECT_DIR/logging/cluster-logs"
        tmux kill-session -t "$session" 2>/dev/null || true
        tmux new -d -s "$session" "bash -lc 'cd \"$PROJECT_DIR\" && $run_cmd 2>&1 | tee \"logging/cluster-logs/${session}__${host}.log\"; exec bash'"
        ok "Launched $session on local host $host"
    else
        ssh -o StrictHostKeyChecking=no "$host" "bash -lc 'mkdir -p $REMOTE_PROJECT_DIR/logging/cluster-logs && tmux kill-session -t $session 2>/dev/null || true && tmux new -d -s $session \"bash -lc '\''cd $REMOTE_PROJECT_DIR && $run_cmd 2>&1 | tee logging/cluster-logs/${session}__${host}.log; exec bash'\''\"'"
        ok "Launched $session on remote host $host"
    fi
}

# ── Build unique host list ─────────────────────────────────────────────────────

ALL_HOSTS=("$MASTER_HOST")
for worker in "${WORKER_ENTRIES[@]}"; do
    worker_host="${worker%%:*}"; worker_rank="${worker##*:}"
    rank_selected "$worker_rank" && ALL_HOSTS+=("$worker_host")
done

UNIQUE_HOSTS=()
for host in "${ALL_HOSTS[@]}"; do
    skip=false
    for existing in "${UNIQUE_HOSTS[@]+"${UNIQUE_HOSTS[@]}"}"; do
        [[ "$existing" == "$host" ]] && { skip=true; break; }
    done
    [[ "$skip" == "false" ]] && UNIQUE_HOSTS+=("$host")
done

info "${C_BOLD}Project:${C_RESET} $PROJECT_DIR"
info "${C_BOLD}Config :${C_RESET} $CONFIG_FILE"
info "${C_BOLD}Master :${C_RESET} $MASTER_HOST"
info "${C_BOLD}Workers:${C_RESET} ${WORKER_ENTRIES[*]}"

# ── Rsync latest code ──────────────────────────────────────────────────────────

info "Syncing code to remote hosts..."
for host in "${UNIQUE_HOSTS[@]}"; do
    if is_local_host "$host"; then
        warn "  $host is local - skipping rsync"
        continue
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        warn "  [DRY RUN] rsync -> $host:$REMOTE_PROJECT_DIR"
        continue
    fi
    info "  rsync -> $host"
    rsync -az \
        --exclude '.git' \
        --exclude '.venv' \
        --exclude 'received_model' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude 'test/fixtures' \
        --exclude 'shards' \
        --exclude 'logging/cluster-logs' \
        "$PROJECT_DIR/" "$host:$REMOTE_PROJECT_DIR/"
done

# ── Kill stale sessions & free ports ──────────────────────────────────────────

if [[ "$DRY_RUN" != "true" ]]; then
    info "Cleaning previous tmux sessions..."
    if is_local_host "$MASTER_HOST"; then
        tmux kill-session -t syncps_api 2>/dev/null || true
    else
        ssh -o StrictHostKeyChecking=no "$MASTER_HOST" "tmux kill-session -t syncps_api 2>/dev/null || true"
    fi

    for worker in "${WORKER_ENTRIES[@]}"; do
        worker_host="${worker%%:*}"; worker_rank="${worker##*:}"
        rank_selected "$worker_rank" || continue
        if is_local_host "$worker_host"; then
            tmux kill-session -t "syncps_worker_${worker_rank}" 2>/dev/null || true
            lsof -ti ":$((9200 + worker_rank))" | xargs kill -9 2>/dev/null || true
        else
            ssh -o StrictHostKeyChecking=no "$worker_host" \
                "tmux kill-session -t syncps_worker_${worker_rank} 2>/dev/null || true; fuser -k $((9200 + worker_rank))/tcp 2>/dev/null || true"
        fi
    done

    if is_local_host "$MASTER_HOST"; then
        lsof -ti :8000 | xargs kill -9 2>/dev/null || true
        lsof -ti :8001 | xargs kill -9 2>/dev/null || true
    else
        ssh -o StrictHostKeyChecking=no "$MASTER_HOST" "fuser -k 8000/tcp 2>/dev/null || true; fuser -k 8001/tcp 2>/dev/null || true"
    fi
fi

# ── Launch ─────────────────────────────────────────────────────────────────────

if [[ "$API_ONLY" == "true" ]]; then
    info "Mode: --api-only - checking worker heartbeats..."
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
    [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="python3"
    if ! "$PYTHON_BIN" "$PROJECT_DIR/utils/check_workers.py"; then
        err "Worker heartbeat check failed. Not launching API."
        exit 1
    fi
    ok "All workers alive - launching API only."
    launch_on_node "$MASTER_HOST" "syncps_api" \
        "if [[ -x .venv/bin/uvicorn ]]; then .venv/bin/uvicorn backend.api:app --host 0.0.0.0 --port 8000; else uvicorn backend.api:app --host 0.0.0.0 --port 8000; fi"
else
    launch_on_node "$MASTER_HOST" "syncps_api" \
        "if [[ -x .venv/bin/uvicorn ]]; then .venv/bin/uvicorn backend.api:app --host 0.0.0.0 --port 8000; else uvicorn backend.api:app --host 0.0.0.0 --port 8000; fi"
    launch_on_node "$MASTER_HOST" "syncps_watcher" \
        "if [[ -x .venv/bin/python ]]; then .venv/bin/python watcher/watch.py --ext '${WATCH_EXT}'; else python3 watcher/watch.py --ext '${WATCH_EXT}'; fi"

    for worker in "${WORKER_ENTRIES[@]}"; do
        worker_host="${worker%%:*}"; worker_rank="${worker##*:}"
        rank_selected "$worker_rank" || { warn "  Skipping rank ${worker_rank} (not in --workers list)"; continue; }
        launch_on_node "$worker_host" "syncps_worker_${worker_rank}" \
            "if [[ -x .venv/bin/python ]]; then .venv/bin/python algorithms/SyncPS/worker.py ${worker_rank} ${worker_host}; else python3 algorithms/SyncPS/worker.py ${worker_rank} ${worker_host}; fi"
    done
fi

ok "Launch complete."
info "API logs:        ssh $MASTER_HOST 'tmux attach -t syncps_api'"
info "Trigger gather:  python main.py  (or POST http://$MASTER_HOST:8000/gather-shards)"
[[ -n "$WORKER_RANKS" ]] && info "Workers launched: ranks ${WORKER_RANKS} only (others untouched)"
