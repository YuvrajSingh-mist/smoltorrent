"""Entry-point wrapper — puts the project root on sys.path then delegates to main.py."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main() -> None:
    from main import main as entry
    entry()
