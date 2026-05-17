"""Integration tests for _run_pending_loop.

Uses real worker APIs and realistic shard sizes (~150 MB and ~400 MB) to
mirror actual traffic. Requires cluster running (scripts/launch.sh).

Markers: integration
"""
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from networking.send_receive import receive_message, send_message
from utils.shard_ops import request_store_shards
from watcher.watch import _run_pending_loop

_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"


def _load_config() -> dict:
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


CFG     = _load_config()
WORKERS = CFG["devices_config"]["workers"]
CKPT_ROOT = Path(CFG["ckpt_root"]).expanduser()


def _send_recv(worker: dict, msg, timeout: float = 30.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((worker["ip"], worker["port"]))
    sock.settimeout(None)
    send_message(sock, msg)
    result = receive_message(sock)
    sock.close()
    return result


def _workers_have_shard(rel: str) -> bool:
    for w in WORKERS:
        missing = _send_recv(w, ("all_shards_present", w["rank"], [rel]))
        if missing:
            return False
    return True


def _cleanup_rel(rel: str) -> None:
    shutil.rmtree(CKPT_ROOT / rel.split("/")[0], ignore_errors=True)
    for w in WORKERS:
        try:
            subprocess.run(
                ["ssh", w["host"],
                 f"rm -rf ~/Desktop/smoltorrent/shards/worker_{w['rank']}/{rel.split('/')[0]}"],
                timeout=15,
            )
        except Exception:
            pass


def _make_checkpoint(path: Path, target_mb: int) -> None:
    """Write a safetensors checkpoint of ~target_mb using bf16 tensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each tensor: [2048, 2048] bf16 = 8 MB. Repeat to hit target.
    tensor_mb = 8
    n = max(1, target_mb // tensor_mb)
    weights = {f"layer_{i}.weight": mx.ones([2048, 2048], dtype=mx.bfloat16) for i in range(n)}
    mx.save_safetensors(str(path), weights)


def _write_in_background(path: Path, target_mb: int) -> threading.Thread:
    """Start writing the checkpoint in a background thread. Returns the thread."""
    t = threading.Thread(target=_make_checkpoint, args=(path, target_mb), daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _start_pending_loop(pending, lock, trigger):
    t = threading.Thread(target=_run_pending_loop, args=(pending, lock, trigger), daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Test 1: ~150 MB file (LFM shard size) — unstable → stable → transferred
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_pending_promotes_and_transfers_150mb():
    """A ~150 MB file written in the background goes pending (unstable),
    then the pending loop promotes it once writing finishes and triggers
    a real transfer to all workers."""
    rel = "__pytest_pending__/lfm-150mb/latest"
    ckpt_file = CKPT_ROOT / rel / "model.safetensors"
    _cleanup_rel(rel)

    pending = []
    lock    = threading.Lock()
    trigger = threading.Event()

    # Start writing — file will be unstable while write is in progress
    write_thread = _write_in_background(ckpt_file, target_mb=150)

    # Add to pending immediately (mirrors on_created seeing an unstable file)
    # Wait briefly so the file exists but isn't finished
    time.sleep(0.5)
    with lock:
        pending.append(ckpt_file)

    _start_pending_loop(pending, lock, trigger)

    # Wait for write to complete then for pending loop to promote
    write_thread.join(timeout=60)
    assert trigger.wait(timeout=30), "pending loop did not fire trigger after file became stable"
    assert pending == [], "stable file should be removed from pending"

    # Transfer via real API
    request_store_shards(ckpt_path=str(ckpt_file), log_fn=lambda m: None)

    # Poll workers
    deadline = time.time() + 120
    while time.time() < deadline:
        if _workers_have_shard(rel):
            break
        time.sleep(5)
    else:
        _cleanup_rel(rel)
        pytest.fail("Workers did not receive ~150 MB shard within 2 minutes")

    _cleanup_rel(rel)


# ---------------------------------------------------------------------------
# Test 2: ~400 MB file (Qwen shard size) — same flow, heavier
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_pending_promotes_and_transfers_400mb():
    """Same flow as the 150 MB test but with ~400 MB to stress real transfer
    throughput over the Pi network link."""
    rel = "__pytest_pending__/qwen-400mb/latest"
    ckpt_file = CKPT_ROOT / rel / "model.safetensors"
    _cleanup_rel(rel)

    pending = []
    lock    = threading.Lock()
    trigger = threading.Event()

    write_thread = _write_in_background(ckpt_file, target_mb=400)
    time.sleep(0.5)
    with lock:
        pending.append(ckpt_file)

    _start_pending_loop(pending, lock, trigger)

    write_thread.join(timeout=120)
    assert trigger.wait(timeout=30), "pending loop did not fire trigger after 400 MB file stabilised"
    assert pending == [], "stable file should be removed from pending"

    request_store_shards(ckpt_path=str(ckpt_file), log_fn=lambda m: None)

    deadline = time.time() + 300
    while time.time() < deadline:
        if _workers_have_shard(rel):
            break
        time.sleep(10)
    else:
        _cleanup_rel(rel)
        pytest.fail("Workers did not receive ~400 MB shard within 5 minutes")

    _cleanup_rel(rel)


# ---------------------------------------------------------------------------
# Test 3: still-unstable file stays in pending, no premature transfer
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_unstable_file_not_transferred():
    """A file that never finishes writing should stay in pending and never
    reach any worker."""
    rel = "__pytest_pending__/unstable/latest"
    ckpt_file = CKPT_ROOT / rel / "model.safetensors"
    _cleanup_rel(rel)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

    pending = []
    lock    = threading.Lock()
    trigger = threading.Event()

    # Write a tiny placeholder that keeps changing size (open and hold)
    ckpt_file.write_bytes(b"\x00" * 1024)
    with lock:
        pending.append(ckpt_file)

    # Pending loop runs but _is_stable returns False because we keep writing
    def keep_writing():
        # Write every 0.3s — faster than _is_stable's 1s window so size always changes
        for _ in range(40):
            time.sleep(0.3)
            with open(ckpt_file, "ab") as f:
                f.write(b"\x00" * 1024)

    writer = threading.Thread(target=keep_writing, daemon=True)
    writer.start()

    _start_pending_loop(pending, lock, trigger)

    # Pending loop polls every 10s — wait 15s; file is still changing so trigger must not fire
    writer.join(timeout=20)
    time.sleep(3)

    assert not trigger.is_set(), "trigger must not fire while file is still being written"
    assert ckpt_file in pending, "unstable file must remain in pending"
    assert not _workers_have_shard(rel), "unstable file must not have been transferred to workers"

    _cleanup_rel(rel)


# ---------------------------------------------------------------------------
# Test 4: multiple files — mix of stable and unstable
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_mixed_pending_only_stable_transferred():
    """Two files in pending: one stable (150 MB, done writing), one still
    being written. Only the stable one should be promoted and transferred."""
    rel_stable   = "__pytest_pending__/mixed-stable/latest"
    rel_unstable = "__pytest_pending__/mixed-unstable/latest"
    ckpt_stable   = CKPT_ROOT / rel_stable   / "model.safetensors"
    ckpt_unstable = CKPT_ROOT / rel_unstable / "model.safetensors"
    _cleanup_rel("__pytest_pending__")

    # Write the stable file fully first
    _make_checkpoint(ckpt_stable, target_mb=150)

    # Start the unstable file writing (will still be in progress)
    ckpt_unstable.parent.mkdir(parents=True, exist_ok=True)
    ckpt_unstable.write_bytes(b"\x00" * 1024)

    pending = [ckpt_stable, ckpt_unstable]
    lock    = threading.Lock()
    trigger = threading.Event()

    def keep_unstable_writing():
        for _ in range(40):
            time.sleep(0.3)
            with open(ckpt_unstable, "ab") as f:
                f.write(b"\x00" * 1024)

    writer = threading.Thread(target=keep_unstable_writing, daemon=True)
    writer.start()

    _start_pending_loop(pending, lock, trigger)

    assert trigger.wait(timeout=30), "trigger should fire for the stable file"

    # Writer is still running — check unstable state while it's actively being written
    assert ckpt_unstable in pending, "unstable file must still be in pending while being written"
    assert not _workers_have_shard(rel_unstable), \
        "unstable file must not have been transferred while still being written"

    writer.join(timeout=15)

    # Transfer the stable one
    request_store_shards(ckpt_path=str(ckpt_stable), log_fn=lambda m: None)

    deadline = time.time() + 120
    while time.time() < deadline:
        if _workers_have_shard(rel_stable):
            break
        time.sleep(5)
    else:
        _cleanup_rel("__pytest_pending__")
        pytest.fail("Workers did not receive stable shard")

    _cleanup_rel("__pytest_pending__")
