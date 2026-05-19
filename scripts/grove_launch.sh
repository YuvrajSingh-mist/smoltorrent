#!/bin/bash
# Start API + watcher on the master only.
# Workers are already running via `grove join` — no SSH or rsync needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/configs/config.yaml"
WATCH_EXT="${WATCH_EXT:-.safetensors}"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if [[ -t 1 ]]; then
    GREEN="\033[0;32m"; YELLOW="\033[0;33m"; RED="\033[0;31m"; RESET="\033[0m"
else
    GREEN=""; YELLOW=""; RED=""; RESET=""
fi
ok()   { echo -e "${GREEN}[grove]${RESET} $*"; }
warn() { echo -e "${YELLOW}[grove]${RESET} $*"; }
err()  { echo -e "${RED}[grove] Error:${RESET} $*" >&2; }

# ── Preflight ────────────────────────────────────────────────────────────────

OS="$(uname -s)"

# 1. tmux
if ! command -v tmux >/dev/null 2>&1; then
    warn "tmux not found — installing..."
    if [[ "$OS" == "Darwin" ]]; then
        if ! command -v brew >/dev/null 2>&1; then
            err "Homebrew is required to install tmux on macOS. Install from https://brew.sh then re-run."
            exit 1
        fi
        brew install tmux
    elif [[ "$OS" == "Linux" ]]; then
        sudo apt update && sudo apt install -y tmux
    else
        err "Unsupported OS for automatic tmux install: $OS"
        exit 1
    fi
fi

# 2. uv
if ! command -v uv >/dev/null 2>&1; then
    warn "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        err "uv installed but not found in PATH — open a new shell and re-run."
        exit 1
    fi
fi

# 3. .venv / uv sync
if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
    warn ".venv not found — running uv sync..."
    (cd "$PROJECT_DIR" && uv sync)
fi

UV="$PROJECT_DIR/.venv/bin"

if [[ ! -x "$UV/uvicorn" ]]; then
    warn "uvicorn missing from .venv — running uv sync..."
    (cd "$PROJECT_DIR" && uv sync)
    if [[ ! -x "$UV/uvicorn" ]]; then
        err "uvicorn still not found after uv sync. Check pyproject.toml dependencies."
        exit 1
    fi
fi

# 4. node_exporter (metrics — non-fatal, just warn)
if ! command -v node_exporter >/dev/null 2>&1; then
    warn "node_exporter not found — system metrics (CPU/disk/memory) will be missing from Grafana."
    warn "Install with: brew install node_exporter  (then run: bash scripts/launch.sh --daemons)"
fi

# 5. config.yaml exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    err "configs/config.yaml not found."
    err "Run 'grove start -n N' first — it writes the config once all workers have joined."
    exit 1
fi

# 6. config.yaml has at least one worker
worker_count="$("$UV/python" - "$CONFIG_FILE" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
workers = (cfg.get("devices_config") or {}).get("workers") or []
print(len(workers))
PYEOF
)"
if [[ "$worker_count" -eq 0 ]]; then
    err "configs/config.yaml has no workers."
    err "Run 'grove start -n N' and wait for all workers to join before launching."
    exit 1
fi
ok "Config OK — $worker_count worker(s) registered"

# 7. Free ports before launch
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8001/tcp 2>/dev/null || true

# ── Launch ───────────────────────────────────────────────────────────────────

mkdir -p "$PROJECT_DIR/logging/cluster-logs"

tmux kill-session -t syncps_api 2>/dev/null || true
tmux new -d -s syncps_api \
    "bash -lc 'cd \"$PROJECT_DIR\" && $UV/uvicorn backend.api:app --host 0.0.0.0 --port 8000 2>&1 | tee logging/cluster-logs/syncps_api__localhost.log; exec bash'"
ok "API started      → tmux attach -t syncps_api"

tmux kill-session -t syncps_watcher 2>/dev/null || true
tmux new -d -s syncps_watcher \
    "bash -lc 'cd \"$PROJECT_DIR\" && $UV/python watcher/watch.py --ext \"$WATCH_EXT\" 2>&1 | tee logging/cluster-logs/syncps_watcher__localhost.log; exec bash'"
ok "Watcher started  → tmux attach -t syncps_watcher"

warn "Workers already running via grove join — attach to a worker and check tmux if needed."
