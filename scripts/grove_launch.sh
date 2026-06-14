#!/bin/bash
# The brain: starts API + watcher on the master.
# Workers join separately via `grove join` on each node.
#
# Usage: grove_launch.sh
#   WATCH_EXT=".safetensors,.pth" grove_launch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/configs/dev-config.yaml"
WATCH_EXT="${WATCH_EXT:-.safetensors}"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    GREEN="\033[0;32m"; YELLOW="\033[0;33m"; RED="\033[0;31m"; RESET="\033[0m"
else
    GREEN=""; YELLOW=""; RED=""; RESET=""
fi
ok()   { echo -e "${GREEN}[grove]${RESET} $*"; }
warn() { echo -e "${YELLOW}[grove]${RESET} $*"; }
err()  { echo -e "${RED}[grove] Error:${RESET} $*" >&2; }

# ── Preflight ────────────────────────────────────────────────────────────────

if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
    err "No .venv found — run bootstrap first: bash scripts/bootstrap.sh"
    exit 1
fi

UV="$PROJECT_DIR/.venv/bin"

if [[ ! -x "$UV/uvicorn" ]]; then
    err "uvicorn missing from .venv — run: cd $PROJECT_DIR && uv sync"
    exit 1
fi

if ! command -v node_exporter >/dev/null 2>&1; then
    warn "node_exporter not found — system metrics will be missing from Grafana."
    warn "Run bash scripts/bootstrap.sh to install it."
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    err "configs/config.yaml not found — run 'grove start -n N' first."
    exit 1
fi

worker_count="$("$UV/python" - "$CONFIG_FILE" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(len((cfg.get("devices_config") or {}).get("workers") or []))
PYEOF
)"
if [[ "$worker_count" -eq 0 ]]; then
    err "configs/config.yaml has no workers — run 'grove start -n N' and wait for workers to join."
    exit 1
fi
ok "Config OK — $worker_count worker(s) registered"

# Kill all local tmux sessions except server-0, then free ports
tmux list-sessions -F '#{session_name}' 2>/dev/null \
    | grep -v '^server-0$' \
    | xargs -I{} tmux kill-session -t {} 2>/dev/null || true
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :8001 | xargs kill -9 2>/dev/null || true

# ── Launch API + watcher ─────────────────────────────────────────────────────

mkdir -p "$PROJECT_DIR/logging/cluster-logs"

# Prevent tmux server from dying when sessions exit
tmux set-option -g remain-on-exit on 2>/dev/null || true

tmux new -d -s grove_api \
    "bash -lc 'cd \"$PROJECT_DIR\" && $UV/uvicorn backend.api:app --host 0.0.0.0 --port 8000 2>&1 | tee logging/cluster-logs/grove_api.log; echo \"[grove_api] uvicorn exited (code \$?)\"; while sleep 3600; do :; done'"
ok "API started      → tmux attach -t grove_api"

tmux new -d -s grove_watcher \
    "bash -lc 'cd \"$PROJECT_DIR\" && $UV/python watcher/watch.py --ext \"$WATCH_EXT\" 2>&1 | tee logging/cluster-logs/grove_watcher.log; echo \"[grove_watcher] watcher exited (code \$?)\"; while sleep 3600; do :; done'"
ok "Watcher started  → tmux attach -t grove_watcher"
ok "API + watcher running.  Trigger: POST http://localhost:8000/gather-shards"
