"""Entry-point wrapper — puts the project root on sys.path then delegates to main.py."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def main() -> None:
    from main import main as _main
    _main()
