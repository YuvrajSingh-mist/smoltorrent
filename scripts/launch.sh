#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/configs/config.yaml"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/Desktop/smoltorrent}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run]"
            exit 1
            ;;
    esac
done

if ! command -v yq >/dev/null 2>&1; then
    echo "Error: yq is required to parse $CONFIG_FILE"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config file not found: $CONFIG_FILE"
    exit 1
fi

MASTER_HOST="$(yq '.devices_config.master[0].host // .devices_config.master.host' "$CONFIG_FILE")"
if [[ -z "$MASTER_HOST" || "$MASTER_HOST" == "null" ]]; then
    echo "Error: could not read master host from $CONFIG_FILE"
    exit 1
fi

WORKER_ENTRIES=()
while IFS= read -r entry; do
    [[ -n "$entry" && "$entry" != "null" ]] && WORKER_ENTRIES+=("$entry")
done < <(yq '.devices_config.workers[] | (.device // .host) + ":" + (.rank | tostring)' "$CONFIG_FILE")

if [[ ${#WORKER_ENTRIES[@]} -eq 0 ]]; then
    echo "Error: no workers found in $CONFIG_FILE"
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

    echo "Installing uv locally..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: uv installed locally but not found in PATH"
        return 1
    fi
}

ensure_local_dependencies() {
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

    if ! command -v tmux >/dev/null 2>&1; then
        echo "Local host is missing tmux. Installing..."
        if [[ "$(uname -s)" == "Darwin" ]]; then
            if ! command -v brew >/dev/null 2>&1; then
                echo "Error: Homebrew is required to install tmux on local macOS host"
                return 1
            fi
            brew install tmux
        elif [[ "$(uname -s)" == "Linux" ]]; then
            sudo apt update && sudo apt install -y tmux curl ca-certificates
        else
            echo "Error: unsupported local OS for automatic tmux install: $(uname -s)"
            return 1
        fi
    fi

    install_uv_local

    if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
        echo "Creating local .venv with uv..."
        (cd "$PROJECT_DIR" && uv venv --python 3.10 .venv)
    fi
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
EOF
}

launch_on_node() {
    local host="$1"
    local session="$2"
    local run_cmd="$3"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] $host :: $session -> $run_cmd"
        return 0
    fi

    if is_local_host "$host"; then
        if ! ensure_local_dependencies; then
            echo "Error: failed to prepare local dependencies on $host"
            return 127
        fi
        mkdir -p "$PROJECT_DIR/logging/cluster-logs"
        tmux kill-session -t "$session" 2>/dev/null || true
        tmux new -d -s "$session" "bash -lc 'cd \"$PROJECT_DIR\" && $run_cmd 2>&1 | tee \"logging/cluster-logs/${session}__${host}.log\"; exec bash'"
        echo "Launched $session on local host $host"
    else
        if ! ensure_remote_dependencies "$host"; then
            echo "Error: failed to prepare remote dependencies on host $host"
            return 127
        fi
        ssh -o StrictHostKeyChecking=no "$host" "bash -lc 'mkdir -p $REMOTE_PROJECT_DIR/logging/cluster-logs && tmux kill-session -t $session 2>/dev/null || true && tmux new -d -s $session \"bash -lc '\''cd $REMOTE_PROJECT_DIR && $run_cmd 2>&1 | tee logging/cluster-logs/${session}__${host}.log; exec bash'\''\"'"
        echo "Launched $session on remote host $host"
    fi
}

echo "Project: $PROJECT_DIR"
echo "Config : $CONFIG_FILE"
echo "Master : $MASTER_HOST"
echo "Workers: ${WORKER_ENTRIES[*]}"

# Remove stale SyncPS sessions first.
if [[ "$DRY_RUN" != "true" ]]; then
    if is_local_host "$MASTER_HOST"; then
        tmux kill-session -t syncps_server 2>/dev/null || true
    else
        ssh -o StrictHostKeyChecking=no "$MASTER_HOST" "tmux kill-session -t syncps_server 2>/dev/null || true"
    fi

    for worker in "${WORKER_ENTRIES[@]}"; do
        worker_host="${worker%%:*}"
        worker_rank="${worker##*:}"
        if is_local_host "$worker_host"; then
            tmux kill-session -t "syncps_worker_${worker_rank}" 2>/dev/null || true
        else
            ssh -o StrictHostKeyChecking=no "$worker_host" "tmux kill-session -t syncps_worker_${worker_rank} 2>/dev/null || true"
        fi
    done
fi

# Launch server from algorithms/SyncPS
launch_on_node "$MASTER_HOST" "syncps_server" "if [[ -x .venv/bin/python ]]; then .venv/bin/python algorithms/SyncPS/server.py; else python3 algorithms/SyncPS/server.py; fi"

# Launch workers from algorithms/SyncPS
for worker in "${WORKER_ENTRIES[@]}"; do
    worker_host="${worker%%:*}"
    worker_rank="${worker##*:}"
    launch_on_node "$worker_host" "syncps_worker_${worker_rank}" "if [[ -x .venv/bin/python ]]; then .venv/bin/python algorithms/SyncPS/worker.py ${worker_rank}; else python3 algorithms/SyncPS/worker.py ${worker_rank}; fi"
done

echo "Launch complete."
echo "Attach example: ssh $MASTER_HOST 'tmux attach -t syncps_server'"
