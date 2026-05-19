"""Exposes server/worker boot time as smoltorrent_boot_time_ms on port 9101."""

import platform
import re
import subprocess
import time

from prometheus_client import Gauge, start_http_server

boot_time = Gauge(
    "smoltorrent_boot_time_ms", "Unix timestamp of last OS boot (milliseconds)"
)


def _get_boot_time_ms() -> float:
    if platform.system() == "Darwin":
        out = subprocess.check_output(["sysctl", "-n", "kern.boottime"], text=True)
        m = re.search(r"sec\s*=\s*(\d+)", out)
        return float(m.group(1)) * 1000 if m else 0.0
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("btime"):
                return float(line.split()[1]) * 1000
    return 0.0


if __name__ == "__main__":
    start_http_server(9101)
    while True:
        boot_time.set(_get_boot_time_ms())
        time.sleep(15)
