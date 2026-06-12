"""CLI entry-point for SmolTorrent.

Usage:
    python main.py start -n <N>   — master: advertise, wait for N workers to join
    python main.py join           — worker: find master via TUI, register, start worker.py
    python main.py store --ckpt-path <path>
    python main.py gather --ckpt-path <path>
"""

import argparse
import logging
import os
import socket
import subprocess
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from json import dumps, loads
from pathlib import Path

import yaml

from utils.common_utils import (
    API_BASE,
    fetch_model_metadata,
    load_config,
    model_id_to_dir_name,
)
from utils.shard_ops import request_gather_shards, request_store_shards

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smoltorrent")

CONFIG_PATH = Path(__file__).parent / "configs" / "config.yaml"
LAUNCH_SCRIPT = Path(__file__).parent / "scripts" / "grove_launch.sh"


# ---------------------------------------------------------------------------
# start — master side
# ---------------------------------------------------------------------------


def _write_config(registered: list[dict]) -> None:
    """Rewrite configs/config.yaml with the current worker list.

    Reads existing config, updates ``num_workers`` and ``devices_config.workers``,
    then writes it back with ``"w"`` (full replace — YAML cannot be appended to).
    Called every time a worker registers, so late-joining workers are recorded.
    """
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    cfg["num_workers"] = len(registered)
    cfg["devices_config"]["workers"] = [
        {"host": f"{w['user']}@{w['ip']}", "ip": w["ip"], "rank": w["rank"], "port": w["port"]}
        for w in registered
    ]
    with CONFIG_PATH.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def cmd_start(n: int) -> None:
    """Advertise this node as master, wait for N workers to join, then launch."""
    from discovery.grove._mdns import MasterAdvertiser, REGISTRATION_PORT
    from discovery.grove._utils import setup_grove_logging

    setup_grove_logging()

    registered: list[dict] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = loads(self.rfile.read(length))
            with lock:
                rank = len(registered) + 1
                body["rank"] = rank
                body.setdefault("port", 5000 + rank)
                registered.append(body)
                # Write config.yaml immediately so late-joining workers are recorded
                _write_config(registered)
            logger.info(
                f"  ✓ {body['user']}@{body['ip']}:{body['port']} ({body['hostname']}) → rank {rank}  [{len(registered)}]"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(dumps({"rank": rank, "port": body["port"]}).encode())

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("0.0.0.0", REGISTRATION_PORT), Handler)
    threading.Thread(target=server.serve_forever).start()  # non-daemon keeps process alive

    advertiser = MasterAdvertiser(expected_workers=n)
    logger.info(f"\n  smoltorrent master ready — waiting for {n} worker(s)")
    logger.info("  On each worker node: grove join\n")

    while True:
        with lock:
            if len(registered) >= n:
                break
        time.sleep(0.5)

    logger.info(f"\n✓ {len(registered)} worker(s) registered — configs/config.yaml kept up to date")
    logger.info("\nLaunching cluster… (additional workers can still 'grove join', Ctrl+C to stop)\n")
    subprocess.Popen(["bash", str(LAUNCH_SCRIPT)])


# ---------------------------------------------------------------------------
# join — worker side
# ---------------------------------------------------------------------------


def cmd_join() -> None:
    """Discover a smoltorrent master via TUI, register, then start worker.py."""
    import httpx
    from discovery.grove._mdns import MasterBrowser
    from discovery.grove._utils import get_local_ip, setup_grove_logging
    from discovery.grove.tui import JoinApp

    setup_grove_logging()

    browser = MasterBrowser()
    time.sleep(2.0)  # give mDNS a moment before showing TUI

    app = JoinApp(browser)
    app.run()
    browser.close()

    selected = app.selected_cluster
    if not selected:
        logger.info("No master selected — exiting.")
        return

    master_ip = selected["ip"]
    master_port = selected["port"]
    my_ip = get_local_ip()
    my_hostname = socket.gethostname()
    my_user = os.environ.get("USER") or os.environ.get("LOGNAME") or my_hostname

    logger.info(f"\n  Registering with master at {master_ip}:{master_port}…")
    resp = httpx.post(
        f"http://{master_ip}:{master_port}",
        json={"hostname": my_hostname, "ip": my_ip, "user": my_user},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    rank = data["rank"]
    port = data["port"]
    logger.info(f"  ✓ Joined as rank {rank} on port {port}")

    # Start worker.py in foreground so the user sees its logs
    subprocess.run(
        [sys.executable, "algorithms/SyncPS/worker.py", str(rank), my_hostname, "--port", str(port)]
    )


# ---------------------------------------------------------------------------
# store / gather / argparse
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolTorrent")
    sub = parser.add_subparsers(dest="action", required=True)

    start_p = sub.add_parser(
        "start", help="Master: advertise and wait for workers to join"
    )
    start_p.add_argument(
        "-n",
        type=int,
        required=True,
        metavar="N",
        help="Number of worker nodes to wait for",
    )

    sub.add_parser(
        "join", help="Worker: find master via TUI, register, start worker.py"
    )

    store_p = sub.add_parser("store", help="Shard a checkpoint and push to workers")
    store_p.add_argument("--ckpt-path", required=True, metavar="PATH")

    gather_p = sub.add_parser("gather", help="Pull shards from workers and merge")
    gather_p.add_argument("--ckpt-path", required=True, metavar="PATH")
    gather_p.add_argument("--model-id", metavar="MODEL_ID", default=None)

    args = parser.parse_args()

    if args.action == "start":
        cmd_start(args.n)
    elif args.action == "join":
        cmd_join()
    elif args.action == "store":
        logger.info("[cli] Storing shards for %s...", args.ckpt_path)
        request_store_shards(ckpt_path=args.ckpt_path, log_fn=logger.info)
    else:
        logger.info("[cli] Gathering shards for %s...", args.ckpt_path)
        request_gather_shards(ckpt_path=args.ckpt_path, log_fn=logger.info)
        if args.model_id:
            logger.info("[cli] Fetching tokenizer and config from HuggingFace Hub...")
            config = load_config()
            fetch_model_metadata(args.model_id, config)
            logger.info("[cli] Model directory ready for inference")
        else:
            logger.info("[cli] Done — merged.safetensors is ready")


if __name__ == "__main__":
    main()
