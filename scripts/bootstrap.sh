#!/bin/bash
# One-time cluster setup: rsync code to all Pi workers and install all dependencies
# (uv, tmux, node_exporter, Python venv, zeroconf, boot_exporter service) on every node.
#
# Run this once before your first launch, or after adding a new worker.
# After it completes you can go straight to grove or launch.sh — no further setup needed.
#
# Usage: bootstrap.sh [--workers <rank,...>] [--dry-run]
#
#   --workers 1,3   bootstrap only the specified worker ranks (default: all)
#   --dry-run       print what would run without executing anything
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/configs/config.yaml"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/Desktop/smoltorrent}"
DRY_RUN=false
WORKER_RANKS=""

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
        --dry-run) DRY_RUN=true; shift ;;
        --workers)
            shift
            [[ $# -eq 0 ]] && { err "--workers requires a comma-separated rank list (e.g. --workers 1,3)"; exit 1; }
            WORKER_RANKS="$1"; shift ;;
        *)
            err "Unknown option: $1"
            warn "Usage: $0 [--workers <rank,...>] [--dry-run]"
            exit 1 ;;
    esac
done

# ── Preflight ──────────────────────────────────────────────────────────────────

if ! command -v yq >/dev/null 2>&1; then
    err "yq is required to parse $CONFIG_FILE  (brew install yq)"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    err "Config file not found: $CONFIG_FILE"
    err "Edit configs/config.yaml with your cluster topology before running bootstrap."
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

install_uv_local() {
    if command -v uv >/dev/null 2>&1; then return 0; fi
    info "Installing uv locally..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        err "uv installed but not found in PATH"
        return 1
    fi
}

ensure_local_dependencies() {
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
    local os; os="$(uname -s)"

    if ! command -v tmux >/dev/null 2>&1; then
        warn "tmux not found — installing..."
        if [[ "$os" == "Darwin" ]]; then
            command -v brew >/dev/null 2>&1 || { err "Homebrew required to install tmux"; return 1; }
            brew install tmux
        elif [[ "$os" == "Linux" ]]; then
            sudo apt update && sudo apt install -y tmux curl ca-certificates
        else
            err "Unsupported OS for automatic tmux install: $os"; return 1
        fi
    fi

    if ! command -v node_exporter >/dev/null 2>&1; then
        warn "node_exporter not found — installing..."
        if [[ "$os" == "Darwin" ]]; then
            command -v brew >/dev/null 2>&1 || { err "Homebrew required to install node_exporter"; return 1; }
            brew install node_exporter
            warn "Run 'bash scripts/launch.sh --daemons' to register node_exporter for auto-start on boot."
        elif [[ "$os" == "Linux" ]]; then
            sudo apt update && sudo apt install -y prometheus-node-exporter
            sudo systemctl enable --now prometheus-node-exporter
            ok "node_exporter installed and enabled via systemd (port 9100)"
        else
            err "Unsupported OS for automatic node_exporter install: $os"; return 1
        fi
    else
        ok "node_exporter already installed"
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
    if command -v uv >/dev/null 2>&1; then return 0; fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
    command -v uv >/dev/null 2>&1
}

_os="$(uname -s)"

if ! command -v tmux >/dev/null 2>&1; then
    echo "Installing tmux on $(hostname)..."
    case "$_os" in
        Darwin)
            command -v brew >/dev/null 2>&1 || { echo "Error: Homebrew required on $(hostname)"; exit 1; }
            brew install tmux ;;
        Linux)
            sudo apt update && sudo apt install -y tmux curl ca-certificates ;;
        *)
            echo "Error: unsupported OS for tmux install: $_os"; exit 1 ;;
    esac
fi

if ! command -v node_exporter >/dev/null 2>&1 && ! systemctl is-active --quiet prometheus-node-exporter 2>/dev/null; then
    echo "Installing node_exporter on $(hostname)..."
    case "$_os" in
        Darwin)
            brew install prometheus-node-exporter && brew services start prometheus-node-exporter ;;
        Linux)
            sudo apt update && sudo apt install -y prometheus-node-exporter
            sudo systemctl enable --now prometheus-node-exporter ;;
        *)
            echo "Warning: cannot auto-install node_exporter on $(hostname)" ;;
    esac
    echo "node_exporter installed on $(hostname) (port 9100)"
else
    echo "node_exporter already present on $(hostname)"
fi

install_uv || { echo "Error: failed to install uv on $(hostname)"; exit 1; }

resolved_project_dir=$(eval echo "$REMOTE_PROJECT_DIR")
mkdir -p "$resolved_project_dir"
cd "$resolved_project_dir"

[[ ! -d .venv ]] && uv venv --python 3.10 .venv

echo "Running uv sync on $(hostname)..."
uv sync

if ! .venv/bin/python -c "import zeroconf" 2>/dev/null; then
    echo "Installing zeroconf on $(hostname)..."
    uv pip install -q zeroconf
    echo "zeroconf installed on $(hostname)"
else
    echo "zeroconf already present on $(hostname)"
fi

# boot_exporter systemd service
BOOT_SERVICE="/etc/systemd/system/smoltorrent-boot-exporter.service"
UV_BIN="$(command -v uv || echo $HOME/.local/bin/uv)"
cat <<SERVICE | sudo tee "$BOOT_SERVICE" > /dev/null
[Unit]
Description=smoltorrent boot time exporter (port 9101)
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$resolved_project_dir
ExecStart=$UV_BIN run $resolved_project_dir/utils/boot_exporter.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE
sudo systemctl daemon-reload
sudo systemctl enable --now smoltorrent-boot-exporter
echo "boot_exporter service registered on $(hostname) (port 9101)"
EOF
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

info "${C_BOLD}Project :${C_RESET} $PROJECT_DIR"
info "${C_BOLD}Config  :${C_RESET} $CONFIG_FILE"
info "${C_BOLD}Master  :${C_RESET} $MASTER_HOST"
info "${C_BOLD}Workers :${C_RESET} ${WORKER_ENTRIES[*]}"

# ── Rsync ──────────────────────────────────────────────────────────────────────

info "Syncing code to remote hosts..."
for host in "${UNIQUE_HOSTS[@]}"; do
    if is_local_host "$host"; then
        warn "  $host is local — skipping rsync"
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

# ── Install dependencies ───────────────────────────────────────────────────────

info "Installing dependencies on all hosts..."
for host in "${UNIQUE_HOSTS[@]}"; do
    if [[ "$DRY_RUN" == "true" ]]; then
        warn "  [DRY RUN] install deps on $host"
        continue
    fi
    info "  Bootstrapping $host..."
    if is_local_host "$host"; then
        ensure_local_dependencies || { err "Failed to bootstrap local host"; exit 127; }
    else
        ensure_remote_dependencies "$host" || { err "Failed to bootstrap $host"; exit 127; }
    fi
    ok "  $host ready"
done

ok ""
ok "Bootstrap complete. All nodes have deps installed."
info ""
info "Next steps — pick one:"
info "  grove (no-SSH):   grove start -n ${#WORKER_ENTRIES[@]}   (master)"
info "                    grove join                               (each worker)"
info "  SSH-based:        bash scripts/launch.sh"
