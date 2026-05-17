#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/configs/config.yaml"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/Desktop/smoltorrent}"
DRY_RUN=false
API_ONLY=false
DAEMONS=false
WORKER_RANKS=""
WATCH_EXT=".safetensors"  # comma-separated extensions for the watcher

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    C_RESET="\033[0m"
    C_BOLD="\033[1m"
    C_RED="\033[31m"
    C_GREEN="\033[32m"
    C_YELLOW="\033[33m"
    C_BLUE="\033[34m"
else
    C_RESET=""
    C_BOLD=""
    C_RED=""
    C_GREEN=""
    C_YELLOW=""
    C_BLUE=""
fi

info() { echo -e "${C_BLUE}${1}${C_RESET}"; }
ok() { echo -e "${C_GREEN}${1}${C_RESET}"; }
warn() { echo -e "${C_YELLOW}${1}${C_RESET}"; }
err() { echo -e "${C_RED}${1}${C_RESET}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --api-only)
            API_ONLY=true
            shift
            ;;
        --daemons)
            DAEMONS=true
            shift
            ;;
        --ext)
            shift
            [[ $# -eq 0 ]] && { err "--ext requires a value (e.g. --ext .safetensors,.pth)"; exit 1; }
            WATCH_EXT="$1"
            shift
            ;;
        --workers)
            shift
            [[ $# -eq 0 ]] && { err "--workers requires a comma-separated rank list (e.g. --workers 1,3)"; exit 1; }
            WORKER_RANKS="$1"
            shift
            ;;
        *)
            err "Unknown option: $1"
            warn "Usage: $0 [--dry-run] [--api-only] [--daemons] [--workers <rank,...>]"
            exit 1
            ;;
    esac
done

if [[ "$DAEMONS" == "true" ]]; then
    STARTUP_SCRIPT="$SCRIPT_DIR/smoltorrent_startup.sh"
    PLIST_LABEL="com.smoltorrent.startup"
    PLIST_DST="/Library/LaunchDaemons/${PLIST_LABEL}.plist"
    SCRIPT_DST="/usr/local/bin/smoltorrent_startup.sh"

    # macOS 26 Tahoe notes:
    #   - launchctl load          → SIGABRT exit 134 (API removed)
    #   - launchctl bootstrap gui → error 125 (GUI domain broken in beta)
    #   - ~/Library/LaunchAgents  → silently ignored (needs SMAppService from Swift)
    #   - /Library/LaunchDaemons  + sudo launchctl enable + bootstrap system → WORKS
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
    # Bootout first in case a stale registration exists (Bootstrap failed: 5 fix)
    sudo launchctl bootout system/"$PLIST_LABEL" 2>/dev/null || true
    sudo launchctl bootstrap system "$PLIST_DST"
    sudo launchctl enable system/"$PLIST_LABEL"

    ok "LaunchDaemon registered — smoltorrent will auto-start on next boot."
    info "Verify:  sudo launchctl print system/${PLIST_LABEL}"
    info "Logs:    tail -f /tmp/smoltorrent-startup.log"
    exit 0
fi

if ! command -v yq >/dev/null 2>&1; then
    err "Error: yq is required to parse $CONFIG_FILE"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    err "Error: config file not found: $CONFIG_FILE"
    exit 1
fi

MASTER_HOST="$(yq '.devices_config.master[0].host // .devices_config.master.host' "$CONFIG_FILE")"
if [[ -z "$MASTER_HOST" || "$MASTER_HOST" == "null" ]]; then
    err "Error: could not read master host from $CONFIG_FILE"
    exit 1
fi

WORKER_ENTRIES=()
while IFS= read -r entry; do
    [[ -n "$entry" && "$entry" != "null" ]] && WORKER_ENTRIES+=("$entry")
done < <(yq '.devices_config.workers[] | (.device // .host) + ":" + (.rank | tostring)' "$CONFIG_FILE")

if [[ ${#WORKER_ENTRIES[@]} -eq 0 ]]; then
    err "Error: no workers found in $CONFIG_FILE"
    exit 1
fi

is_local_host() {
    local host="$1"
    local short_host
    local full_host
    short_host="$(hostname -s)"
    full_host="$(hostname)"
    [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$short_host" || "$host" == "$full_host" ]]
}

install_uv_local() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi

    info "Installing uv locally..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        err "Error: uv installed locally but not found in PATH"
        return 1
    fi
}

ensure_local_dependencies() {
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

    if ! command -v tmux >/dev/null 2>&1; then
        warn "Local host is missing tmux. Installing..."
        if [[ "$(uname -s)" == "Darwin" ]]; then
            if ! command -v brew >/dev/null 2>&1; then
                err "Error: Homebrew is required to install tmux on local macOS host"
                return 1
            fi
            brew install tmux
        elif [[ "$(uname -s)" == "Linux" ]]; then
            sudo apt update && sudo apt install -y tmux curl ca-certificates
        else
            err "Error: unsupported local OS for automatic tmux install: $(uname -s)"
            return 1
        fi
    fi

    install_uv_local

    if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
        info "Creating local .venv with uv..."
        (cd "$PROJECT_DIR" && uv venv --python 3.10 .venv)
    fi

    info "Running local uv sync..."
    (cd "$PROJECT_DIR" && uv sync)
}

ensure_remote_dependencies() {
    local host="$1"

    ssh -o StrictHostKeyChecking=no "$host" "REMOTE_PROJECT_DIR='$REMOTE_PROJECT_DIR' bash -s" <<'EOF'
set -euo pipefail

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
    command -v uv >/dev/null 2>&1
}

if ! command -v tmux >/dev/null 2>&1; then
    echo "Installing tmux on $(hostname)..."
    case "$(uname -s)" in
        Darwin)
            if ! command -v brew >/dev/null 2>&1; then
                echo "Error: Homebrew is required on remote macOS host $(hostname)"
                exit 1
            fi
            brew install tmux
            ;;
        Linux)
            sudo apt update
            sudo apt install -y tmux curl ca-certificates
            ;;
        *)
            echo "Error: unsupported remote OS for automatic tmux install: $(uname -s)"
            exit 1
            ;;
    esac
fi

if ! install_uv; then
    echo "Error: failed to install uv on $(hostname)"
    exit 1
fi

resolved_project_dir=$(eval echo "$REMOTE_PROJECT_DIR")
mkdir -p "$resolved_project_dir"
cd "$resolved_project_dir"

if [[ ! -d .venv ]]; then
    uv venv --python 3.10 .venv
fi

echo "Running remote uv sync on $(hostname)..."
uv sync
EOF
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

rank_selected() {
    local rank="$1"
    [[ -z "$WORKER_RANKS" ]] && return 0
    local IFS=','
    for r in $WORKER_RANKS; do
        [[ "$r" == "$rank" ]] && return 0
    done
    return 1
}

info "${C_BOLD}Project:${C_RESET} $PROJECT_DIR"
info "${C_BOLD}Config :${C_RESET} $CONFIG_FILE"
info "${C_BOLD}Master :${C_RESET} $MASTER_HOST"
info "${C_BOLD}Workers:${C_RESET} ${WORKER_ENTRIES[*]}"

# Build unique host list and sync code to remote hosts first (like smolcluster).
ALL_HOSTS=("$MASTER_HOST")
for worker in "${WORKER_ENTRIES[@]}"; do
    worker_host="${worker%%:*}"
    worker_rank="${worker##*:}"
    rank_selected "$worker_rank" && ALL_HOSTS+=("$worker_host")
done

UNIQUE_HOSTS=()
for host in "${ALL_HOSTS[@]}"; do
    skip=false
    if [[ ${#UNIQUE_HOSTS[@]} -gt 0 ]]; then
        for existing in "${UNIQUE_HOSTS[@]}"; do
            if [[ "$existing" == "$host" ]]; then
                skip=true
                break
            fi
        done
    fi
    if [[ "$skip" == "false" ]]; then
        UNIQUE_HOSTS+=("$host")
    fi
done

info "Syncing code to remote hosts..."
for host in "${UNIQUE_HOSTS[@]}"; do
    if is_local_host "$host"; then
        warn "  $host is local, skipping rsync"
        continue
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        warn "  [DRY RUN] rsync project to $host:$REMOTE_PROJECT_DIR"
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

info "Preparing dependencies on all hosts..."
info "Cleanup phase is deferred until dependency prep completes on every host."
for host in "${UNIQUE_HOSTS[@]}"; do
    if [[ "$DRY_RUN" == "true" ]]; then
        warn "  [DRY RUN] prepare dependencies on $host"
        continue
    fi

    info "Preparing dependencies on host: $host"
    if is_local_host "$host"; then
        if ! ensure_local_dependencies; then
            err "Error: failed to prepare local dependencies on $host"
            exit 127
        fi
    else
        if ! ensure_remote_dependencies "$host"; then
            err "Error: failed to prepare remote dependencies on host $host"
            exit 127
        fi
    fi
    ok "Dependencies ready on host: $host"
done

# Remove stale SyncPS sessions first.
if [[ "$DRY_RUN" != "true" ]]; then
    info "Cleaning previous tmux sessions before launch..."
    if is_local_host "$MASTER_HOST"; then
        info "Cleaning local tmux sessions: syncps_api (host: $MASTER_HOST)"
        tmux kill-session -t syncps_api 2>/dev/null || true
    else
        info "Cleaning remote tmux sessions: syncps_api (host: $MASTER_HOST)"
        ssh -o StrictHostKeyChecking=no "$MASTER_HOST" "tmux kill-session -t syncps_api 2>/dev/null || true"
    fi

    for worker in "${WORKER_ENTRIES[@]}"; do
        worker_host="${worker%%:*}"
        worker_rank="${worker##*:}"
        rank_selected "$worker_rank" || continue
        if is_local_host "$worker_host"; then
            info "Cleaning local tmux session: syncps_worker_${worker_rank} (host: $worker_host)"
            tmux kill-session -t "syncps_worker_${worker_rank}" 2>/dev/null || true
        else
            info "Cleaning remote tmux session: syncps_worker_${worker_rank} (host: $worker_host)"
            ssh -o StrictHostKeyChecking=no "$worker_host" "tmux kill-session -t syncps_worker_${worker_rank} 2>/dev/null || true"
        fi
    done
fi

if [[ "$API_ONLY" == "true" ]]; then
    info "Mode: --api-only — checking worker heartbeats before launching API..."
    PYTHON_BIN="python3"
    if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
        PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
    fi
    if ! "$PYTHON_BIN" "$PROJECT_DIR/utils/check_workers.py"; then
        err "Worker heartbeat check failed. Not launching API."
        exit 1
    fi
    ok "All workers alive — launching API only."
    launch_on_node "$MASTER_HOST" "syncps_api" "if [[ -x .venv/bin/uvicorn ]]; then .venv/bin/uvicorn backend.api:app --host 0.0.0.0 --port 8000; else uvicorn backend.api:app --host 0.0.0.0 --port 8000; fi"
else
    # Launch the shard API on the master
    launch_on_node "$MASTER_HOST" "syncps_api" "if [[ -x .venv/bin/uvicorn ]]; then .venv/bin/uvicorn backend.api:app --host 0.0.0.0 --port 8000; else uvicorn backend.api:app --host 0.0.0.0 --port 8000; fi"
    launch_on_node "$MASTER_HOST" "syncps_watcher" "if [[ -x .venv/bin/python ]]; then .venv/bin/python watcher/watch.py --ext '${WATCH_EXT}'; else python3 watcher/watch.py --ext '${WATCH_EXT}'; fi"

    # Launch workers (all, or only the ranks specified with --workers)
    for worker in "${WORKER_ENTRIES[@]}"; do
        worker_host="${worker%%:*}"
        worker_rank="${worker##*:}"
        rank_selected "$worker_rank" || { warn "  Skipping rank ${worker_rank} (not in --workers list)"; continue; }
        launch_on_node "$worker_host" "syncps_worker_${worker_rank}" "if [[ -x .venv/bin/python ]]; then .venv/bin/python algorithms/SyncPS/worker.py ${worker_rank} ${worker_host}; else python3 algorithms/SyncPS/worker.py ${worker_rank} ${worker_host}; fi"
    done
fi

ok "Launch complete."
info "API logs:        ssh $MASTER_HOST 'tmux attach -t syncps_api'"
info "Trigger gather:  python main.py  (or POST http://$MASTER_HOST:8000/gather-shards)"
[[ -n "$WORKER_RANKS" ]] && info "Workers launched: ranks ${WORKER_RANKS} only (others untouched)"
