"""Compile the Swift P2P helper binary."""

import shutil
import subprocess
from pathlib import Path

from .._utils import get_logger

log = get_logger("swift")

BIN_DIR = Path.home() / ".grove" / "bin"
BIN_NAME = "grove-p2p-helper"
SWIFT_SRC = Path(__file__).parent / "p2p_helper.swift"


def binary_path() -> Path:
    """Return the absolute path where the compiled helper binary lives.

    Returns:
        ``~/.grove/bin/grove-p2p-helper``.
    """
    return BIN_DIR / BIN_NAME


def is_available() -> bool:
    """Check whether the Swift compiler (``swiftc``) is on ``$PATH``.

    Returns:
        ``True`` on macOS with Xcode CLT installed, ``False`` otherwise.
    """
    return shutil.which("swiftc") is not None


def ensure_compiled() -> Path:
    """Return the path to a compiled (and up-to-date) helper binary.

    Compiles ``p2p_helper.swift`` with ``swiftc -O`` if the binary is
    missing or older than the source.  Cached in ``~/.grove/bin/``.

    Returns:
        Absolute :class:`~pathlib.Path` to the native Mach-O binary.

    Raises:
        RuntimeError: If ``swiftc`` is not installed or compilation
            fails.
    """
    bin_path = binary_path()

    if bin_path.exists() and bin_path.stat().st_mtime >= SWIFT_SRC.stat().st_mtime:
        return bin_path

    if not is_available():
        msg = (
            "Swift compiler (swiftc) not found. "
            "Install Xcode command-line tools: xcode-select --install"
        )
        log.warning("[swiftc] %s", msg)
        raise RuntimeError(msg)

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    log.info("[swiftc] compiling P2P helper: %s -> %s", SWIFT_SRC, bin_path)

    result = subprocess.run(
        ["swiftc", "-O", "-o", str(bin_path), str(SWIFT_SRC)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("[swiftc] compilation failed:\n%s", result.stderr)
        raise RuntimeError(f"Swift compilation failed:\n{result.stderr}")

    log.info("[swiftc] P2P helper compiled successfully")
    return bin_path
