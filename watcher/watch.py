"""Filesystem watcher — auto-distributes new checkpoints to workers as they appear.

Watches ``ckpt_root`` (from config.yaml) recursively. When a new ``.safetensors``
file is stable (not still being written), calls ``/store-shard`` on the API.
Persists the set of already-sent files to ``sent.json`` so restarts don't re-send.
"""
import json
import logging
import sys
import time
from pathlib import Path

import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.shard_ops import request_store_shards

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smoltorrent.watcher")

_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "config.yaml"
_SENT_FILE = Path(__file__).parent / "sent.json"


def _load_sent() -> set:
    if not _SENT_FILE.exists():
        return set()
    with open(_SENT_FILE) as f:
        return set(json.load(f))


def _save_sent(sent: set) -> None:
    with open(_SENT_FILE, "w") as f:
        json.dump(sorted(sent), f)


def _is_stable(path: Path, wait: float = 1.0) -> bool:
    """Return True if the file size hasn't changed after ``wait`` seconds."""
    try:
        size_before = path.stat().st_size
        time.sleep(wait)
        return path.stat().st_size == size_before
    except OSError:
        return False


class CheckpointHandler(FileSystemEventHandler):
    """Watch for new .safetensors files and ship them to workers."""

    def __init__(self, sent: set) -> None:
        self._sent = sent

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix != ".safetensors":
            return
        key = str(path)
        if key in self._sent:
            return
        if not _is_stable(path):
            logger.warning("File not stable yet, skipping: %s", path)
            return
        logger.info("New checkpoint detected: %s", path)
        try:
            request_store_shards(ckpt_path=key, log_fn=logger.info)
            self._sent.add(key)
            _save_sent(self._sent)
            logger.info("Stored and recorded: %s", path)
        except Exception as e:
            logger.error("Failed to store %s: %s", path, e)


def main() -> None:
    with _CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)
    ckpt_root = Path(config["ckpt_root"]).expanduser()
    ckpt_root.mkdir(parents=True, exist_ok=True)

    sent = _load_sent()
    logger.info("Watching %s — %d files already sent", ckpt_root, len(sent))

    handler = CheckpointHandler(sent)
    observer = Observer()
    observer.schedule(handler, str(ckpt_root), recursive=True)
    observer.start()
    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
