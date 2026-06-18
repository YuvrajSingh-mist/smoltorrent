"""Centralized logging for smolcluster — ANSI-coloured console output, per-rank filtering, cluster-wide file logging (setup_logging, setup_cluster_logging), and structured event emitters (emit_smol_event, emit_transport_event) consumed by the dashboard SSE stream."""

import logging
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI colour palette
# ---------------------------------------------------------------------------

RESET = "\033[0m"

LEVEL_COLOURS = {
    logging.DEBUG:    "\033[38;5;244m",  # grey
    logging.INFO:     "\033[38;5;114m",  # soft green
    logging.WARNING:  "\033[38;5;214m",  # amber
    logging.ERROR:    "\033[38;5;203m",  # coral red
    logging.CRITICAL: "\033[1;38;5;196m",  # bold bright red
}

LEVEL_BG = {
    logging.WARNING:  "\033[48;5;52m",   # dark red bg for WARNING badge
    logging.ERROR:    "\033[48;5;52m",   # dark red bg for ERROR badge
    logging.CRITICAL: "\033[48;5;88m",   # deeper red bg for CRITICAL
}

TAG_COLOUR  = "\033[38;5;183m"  # soft lavender — bracketed [tags]
DIM         = "\033[2m"
BOLD        = "\033[1m"
TS_COLOUR   = "\033[38;5;240m"  # dark grey for timestamp
NAME_COLOUR = "\033[38;5;110m"  # steel blue for logger name
CTX_COLOUR  = "\033[1;38;5;183m"  # bold lavender for context prefix
SEP         = f"{DIM}│{RESET}"   # subtle column separator

# ---------------------------------------------------------------------------
# Global log context — set once at process startup via set_log_context()
# ---------------------------------------------------------------------------

CTX: dict[str, str] = {}
CTX_ORDER = ("arch", "algorithm", "role", "hardware")


def _infer_hardware(hostname: str) -> str:
    """Derive a human-readable hardware label from a hostname string.

    Args:
        hostname: Raw hostname of the machine (e.g. ``"rpi4"``, ``"macmini1"``).

    Returns:
        A short label such as ``"RPi"``, ``"Mac Mini"``, ``"Jetson"``,
        ``"MacBook"``, or the original hostname if no pattern matches.
        Returns an empty string if *hostname* is empty.
    """
    if not hostname:
        return ""
    h = hostname.lower()
    if re.match(r"(macmini|mini)\d*$", h):
        return "Mac Mini"
    if re.match(r"jetson\w*$", h):
        return "Jetson"
    if re.match(r"(rpi|raspi|pi)\d*$", h):
        return "RPi"
    if re.match(r"macbook\w*$", h):
        return "MacBook"
    return hostname


def set_log_context(
    *,
    algorithm: str = "",
    arch: str = "",
    role: str = "",
    hardware: str = "",
) -> None:
    """Set global context fields that appear on every log line.

    Call once from the main entry-point after :func:`setup_logging`.
    Fields are shown as ``[arch | algorithm | role | hardware]`` in each line
    by :class:`ColourFormatter`.

    Args:
        algorithm: Short algorithm name, e.g. ``"syncps"`` or ``"api"``.
        arch:      Architecture name, e.g. ``"smoltorrent"``.
        role:      Process role string, e.g. ``"server"`` or ``"worker-2"``.
        hardware:  Hardware label, e.g. ``"RPi"`` or ``"Mac Mini"``.

    Returns:
        None.
    """
    for key, val in (
        ("algorithm", algorithm),
        ("arch", arch),
        ("role", role),
        ("hardware", hardware),
    ):
        if val:
            CTX[key] = val


class ColourFormatter(logging.Formatter):
    """Single-line coloured formatter.

    Format:  HH:MM:SS │ LEVEL    │ [ctx] │ logger.name  message
    Bracketed tags like [syncps], [mdns] are highlighted in lavender.
    Context prefix (role | hardware) shown when set via set_log_context().
    """

    _LEVEL_LABELS = {
        logging.DEBUG:    "DEBUG",
        logging.INFO:     "INFO ",
        logging.WARNING:  "WARN ",
        logging.ERROR:    "ERROR",
        logging.CRITICAL: "CRIT ",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Format a log record as a single ANSI-coloured line.

        Args:
            record: The :class:`logging.LogRecord` to format.

        Returns:
            A fully-formatted string ready for writing to stderr, including
            ANSI colour escape codes, bracketed-tag highlighting, and an
            optional exception traceback appended on a new line.
        """
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, "%H:%M:%S")

        lvl_col  = LEVEL_COLOURS.get(record.levelno, "")
        lvl_bg   = LEVEL_BG.get(record.levelno, "")
        lvl_label = self._LEVEL_LABELS.get(record.levelno, record.levelname[:5])

        if lvl_bg:
            level_str = f"{lvl_bg}{lvl_col}{BOLD} {lvl_label} {RESET}"
        else:
            level_str = f"{lvl_col}{lvl_label}{RESET}"

        msg = re.sub(
            r"(\[[^\]]{1,40}\])",
            rf"{TAG_COLOUR}\1{RESET}",
            record.message,
        )
        # Colour the message itself at warning+
        if record.levelno >= logging.WARNING:
            msg = f"{lvl_col}{msg}{RESET}"

        ctx_parts = [CTX[k] for k in CTX_ORDER if CTX.get(k)]
        ctx = f" {CTX_COLOUR}[{' | '.join(ctx_parts)}]{RESET} {SEP}" if ctx_parts else ""

        name = f"{NAME_COLOUR}{record.name}{RESET}"
        ts   = f"{TS_COLOUR}{record.asctime}{RESET}"

        line = f"{ts} {SEP} {level_str} {SEP}{ctx} {name}  {msg}"

        if record.exc_info:
            line = f"{line}\n{LEVEL_COLOURS.get(logging.ERROR, '')}{self.formatException(record.exc_info)}{RESET}"

        return line


def setup_logging(
    level: int = logging.INFO,
    *,
    force: bool = False,
) -> None:
    """Configure the root logger with a coloured console handler.

    Call once from the main entry-point of each script.
    Subsequent calls are no-ops unless ``force=True``.
    """
    root = logging.getLogger()

    if root.handlers and not force:
        return

    if force:
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColourFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # Quieten noisy third-party loggers
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "filelock",
        "datasets",
        "huggingface_hub",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Cluster logging (file-based, structured)
# ---------------------------------------------------------------------------


class RankFilter(logging.Filter):
    """Attach rank and component fields to every log record."""

    def __init__(self, rank: Optional[int] = None, component: str = "server") -> None:
        """Initialise the filter with a rank and component label.

        Args:
            rank:      Integer worker rank, or ``None`` for the server (stored as -1).
            component: Human-readable role string, e.g. ``"server"`` or ``"worker"``.

        Returns:
            None.
        """
        super().__init__()
        self.rank = rank if rank is not None else -1
        self.component = component

    def filter(self, record: logging.LogRecord) -> bool:
        """Inject ``rank`` and ``component`` attributes into *record* and allow it through.

        Args:
            record: The log record being processed.

        Returns:
            Always ``True`` — this filter never suppresses records.
        """
        record.rank = self.rank
        record.component = self.component
        return True


def setup_cluster_logging(
    logger: logging.Logger,
    component: str,
    rank: Optional[int] = None,
    hostname: Optional[str] = None,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    algorithm: str = "",
    arch: str = "",
) -> None:
    """Add a structured (no-ANSI) file handler to an existing logger.

    The log file is placed under ``logging/cluster-logs/`` in the project root
    (or *log_dir* if supplied).  Duplicate file handlers are silently skipped so
    this function is safe to call multiple times.  Also updates the global
    :func:`set_log_context` fields so the console formatter shows the same context.

    Args:
        logger:    The :class:`logging.Logger` to attach the file handler to.
        component: Process role — ``"server"`` or ``"worker"``; controls the log
                   file name prefix.
        rank:      Worker rank integer (only relevant when ``component != "server"``).
        hostname:  Hostname used in the log file name; defaults to ``"unknown"``.
        log_dir:   Override for the cluster-log directory path.
        level:     Minimum log level for the file handler (default ``INFO``).
        algorithm: Algorithm tag written into every log line, e.g. ``"syncps"``.
        arch:      Architecture tag written into every log line, e.g. ``"smoltorrent"``.

    Returns:
        None.
    """

    def _project_log_dir() -> Path:
        """Return the default cluster-log directory path.

        Args:
            None.

        Returns:
            Absolute :class:`~pathlib.Path` to ``<project_root>/logging/cluster-logs``.
        """
        return Path(__file__).resolve().parents[1] / "logging" / "cluster-logs"

    def _pick_writable(preferred: Optional[str]) -> Path:
        """Return the first writable directory from a prioritised candidate list.

        Args:
            preferred: Caller-supplied path string, or ``None`` to use the default.

        Returns:
            An existing, writable :class:`~pathlib.Path`.

        Raises:
            OSError: If no writable candidate can be found or created.
        """
        default = _project_log_dir()
        for candidate in [
            Path(preferred) if preferred else default,
            default,
            Path.cwd() / "smolcluster-logs",
        ]:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / ".write_probe"
                probe.open("a").close()
                probe.unlink(missing_ok=True)
                return candidate
            except OSError:
                continue
        raise OSError("No writable directory found for cluster logs")

    hardware = _infer_hardware(hostname or "")

    # Set global context so ColourFormatter picks it up on every line
    set_log_context(
        algorithm=algorithm,
        arch=arch,
        role=(
            "server"
            if component == "server"
            else f"worker-{rank}"
            if rank is not None
            else "worker"
        ),
        hardware=hardware,
    )

    algo_prefix = f"{algorithm}-" if algorithm else ""
    log_file = _pick_writable(log_dir) / (
        f"{algo_prefix}server-{hostname or 'unknown'}.log"
        if component == "server"
        else f"{algo_prefix}worker-rank{rank}-{hostname or 'unknown'}.log"
    )

    # Avoid duplicate handlers
    if any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
        for h in logger.handlers
    ):
        return

    logger.addFilter(RankFilter(rank=rank, component=component))

    try:
        fh = logging.FileHandler(log_file, mode="a")
    except PermissionError:
        fallback = _pick_writable(
            str(_project_log_dir().parent / "cluster-logs-fallback")
        )
        fh = logging.FileHandler(fallback / log_file.name, mode="a")

    # Build a clean, human-readable format for the file (no ANSI colours)
    ctx_parts = [
        p
        for p in (
            arch,
            algorithm,
            ("server" if component == "server" else f"worker-{rank}"),
            hardware,
        )
        if p
    ]
    ctx_str = " | ".join(ctx_parts)
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            f"%(asctime)s  %(levelname)-8s  [{ctx_str}]  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)
    logger.info(
        "[log] Logging initialised: %s  [algorithm=%s arch=%s role=%s hardware=%s]",
        log_file,
        algorithm or "?",
        arch or "?",
        component,
        hardware or "?",
    )


