#!/usr/bin/env bash
# Startup wrapper launched by com.smoltorrent.startup LaunchAgent.
# Waits for Tailscale to be reachable, then delegates to launch.sh.
set -euo pipefail

SMOLTORRENT_DIR="/Users/yuvrajsingh1/Desktop/smoltorrent"
LOG=/tmp/smoltorrent-startup.log
TAILSCALE_PROBE="100.68.124.90"   # pi4-1 — first worker to come up
TIMEOUT=300                        # give up after 5 min

exec >> "$LOG" 2>&1
echo "[$(date)] smoltorrent_startup: waiting for Tailscale (${TAILSCALE_PROBE})..."

deadline=$(( $(date +%s) + TIMEOUT ))
until ping -c1 -W1 "$TAILSCALE_PROBE" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[$(date)] smoltorrent_startup: timeout — network not ready, aborting."
        exit 1
    fi
    sleep 5
done

echo "[$(date)] smoltorrent_startup: network ready — launching cluster..."
bash "$SMOLTORRENT_DIR/scripts/launch.sh"
echo "[$(date)] smoltorrent_startup: launch.sh exited $?"
