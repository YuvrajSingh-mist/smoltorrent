#!/bin/bash
# Dev tool: rsync latest code to all worker nodes and/or restart local processes.
# Uses configs/dev-config.yaml for SSH aliases + IPs.
#
# Usage:
#   launch.sh [--dry-run]          — rsync to workers only (default)
#   launch.sh --api                — restart API only (no rsync)
#   launch.sh --watcher            — restart watcher only (no rsync)
#   launch.sh --api --watcher      — restart both (no rsync)
#   launch.sh --dry-run --api      — rsync dry-run + restart API
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/configs/dev-config.yaml"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/Desktop/smoltorrent}"
DRY_RUN=false
RUN_API=false
RUN_WATCHER=false

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
        --dry-run)  DRY_RUN=true;    shift ;;
        --api)      RUN_API=true;    shift ;;
        --watcher)  RUN_WATCHER=true; shift ;;
        *)
            err "Unknown option: $1"
            warn "Usage: $0 [--dry-run] [--api] [--watcher]"
            exit 1 ;;
    esac
done

# ── API / Watcher restart (no rsync needed) ────────────────────────────────────

start_api() {
    info "Restarting API..."
    pkill -f "uvicorn backend.api" 2>/dev/null || true
    sleep 1
    cd "$PROJECT_DIR"
    uv run uvicorn backend.api:app --host 0.0.0.0 --port 8000 &
    sleep 2
    if curl -s http://localhost:8000/ >/dev/null 2>&1 || \
       curl -s http://localhost:8000/metrics/ >/dev/null 2>&1; then
        ok "API up on :8000"
    else
        err "API may have failed to start — check logs"
    fi
}

start_watcher() {
    info "Restarting watcher..."
    pkill -f "watcher/watch.py" 2>/dev/null || true
    sleep 1
    cd "$PROJECT_DIR"
    uv run python watcher/watch.py > /tmp/smoltorrent-watcher.log 2>&1 &
    sleep 2
    if pgrep -f "watcher/watch.py" >/dev/null; then
        ok "Watcher up (metrics on :8001) — logs: /tmp/smoltorrent-watcher.log"
    else
        err "Watcher failed to start:"
        tail -10 /tmp/smoltorrent-watcher.log >&2
    fi
}

if [[ "$RUN_API" == "true" || "$RUN_WATCHER" == "true" ]]; then
    [[ "$RUN_API" == "true" ]]     && start_api
    [[ "$RUN_WATCHER" == "true" ]] && start_watcher
    # Skip rsync unless --dry-run was also passed (makes no sense without hosts)
    [[ "$DRY_RUN" == "true" ]] && warn "[DRY RUN] rsync skipped (--api/--watcher mode)"
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

UNIQUE_HOSTS=()
while IFS= read -r h; do
    [[ -n "$h" && "$h" != "null" ]] && UNIQUE_HOSTS+=("$h")
done < <(yq '.devices_config.workers[] | (.device // .host)' "$CONFIG_FILE" | sort -u)

if [[ ${#UNIQUE_HOSTS[@]} -eq 0 ]]; then
    err "No workers found in $CONFIG_FILE"
    exit 1
fi

info "${C_BOLD}Project:${C_RESET} $PROJECT_DIR"
info "${C_BOLD}Config :${C_RESET} $CONFIG_FILE"
info "${C_BOLD}Hosts  :${C_RESET} ${UNIQUE_HOSTS[*]}"

# ── Rsync latest code ──────────────────────────────────────────────────────────

info "Syncing code to remote hosts..."
for host in "${UNIQUE_HOSTS[@]}"; do
    if [[ "$DRY_RUN" == "true" ]]; then
        warn "  [DRY RUN] rsync -> $host:$REMOTE_PROJECT_DIR"
        continue
    fi
    info "  rsync -> $host"
    rsync -az \
        --exclude '.git' \
        --exclude '.venv' \
        --exclude 'configs/' \
        --exclude 'received_model' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude 'test/fixtures' \
        --exclude 'shards' \
        --exclude 'logging/cluster-logs' \
        "$PROJECT_DIR/" "$host:$REMOTE_PROJECT_DIR/"
    ssh "$host" "tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -v '^server-0$' | xargs -I{} tmux kill-session -t {} 2>/dev/null || true"
    ok "  $host — synced + sessions cleared"
done

[[ "$DRY_RUN" == "true" ]] && { warn "[DRY RUN] rsync complete"; exit 0; }

ok "Rsync complete. Run: bash launch.sh --api --watcher to restart API + watcher."
