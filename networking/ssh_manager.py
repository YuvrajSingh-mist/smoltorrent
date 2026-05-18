"""Manages the smoltorrent-owned block in ~/.ssh/config."""
from pathlib import Path

_SSH_CONFIG = Path.home() / ".ssh" / "config"

_BLOCK_START = "### BEGIN SMOLTORRENT MANAGED — do not edit this block ###"
_BLOCK_END = "### END SMOLTORRENT MANAGED ###"
_WARNING = (
    "# This block is auto-managed by smoltorrent.\n"
    "# Run `python main.py discover` to update. Manual edits will be overwritten.\n"
)


def _build_block(workers: list[dict], username: str, identity_file: str | None) -> str:
    lines = [_BLOCK_START + "\n", _WARNING]
    for w in workers:
        lines.append(f"Host {w['hostname']}\n")
        lines.append(f"    HostName {w['ip']}\n")
        lines.append(f"    User {username}\n")
        if identity_file:
            lines.append(f"    IdentityFile {identity_file}\n")
        lines.append("\n")
    lines.append(_BLOCK_END + "\n")
    return "".join(lines)


def write_ssh_block(workers: list[dict], username: str, identity_file: str | None = None) -> None:
    """Write or replace the smoltorrent managed block in ~/.ssh/config.

    Everything outside the block is left untouched.
    """
    _SSH_CONFIG.parent.mkdir(mode=0o700, exist_ok=True)
    existing = _SSH_CONFIG.read_text() if _SSH_CONFIG.exists() else ""

    # Strip existing managed block if present
    lines = existing.splitlines(keepends=True)
    out: list[str] = []
    inside = False
    for line in lines:
        if line.rstrip() == _BLOCK_START:
            inside = True
            continue
        if line.rstrip() == _BLOCK_END:
            inside = False
            continue
        if not inside:
            out.append(line)

    # Ensure a blank line before the new block
    body = "".join(out).rstrip("\n")
    if body:
        body += "\n\n"

    body += _build_block(workers, username, identity_file)
    _SSH_CONFIG.write_text(body)
    _SSH_CONFIG.chmod(0o600)


def remove_ssh_block() -> bool:
    """Remove the smoltorrent managed block. Returns True if a block was found."""
    if not _SSH_CONFIG.exists():
        return False
    lines = _SSH_CONFIG.read_text().splitlines(keepends=True)
    out: list[str] = []
    inside = False
    found = False
    for line in lines:
        if line.rstrip() == _BLOCK_START:
            inside = True
            found = True
            continue
        if line.rstrip() == _BLOCK_END:
            inside = False
            continue
        if not inside:
            out.append(line)
    if found:
        _SSH_CONFIG.write_text("".join(out))
    return found
