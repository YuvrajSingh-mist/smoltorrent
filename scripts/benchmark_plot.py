"""Generate benchmark charts from timed store/gather results and save to docs/img/.

Usage:
    python scripts/benchmark_plot.py

Reads RESULTS dict defined inline, produces three PNGs:
  docs/img/benchmark_wall_time.png   — grouped bar: wall-clock time by model
  docs/img/benchmark_throughput.png  — grouped bar: aggregate MB/s by model
  docs/img/benchmark_speedup.png     — bar: estimated speedup vs old sequential
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path(__file__).parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

# ── Results (fill in after runs) ─────────────────────────────────────────────
# Each list is wall-clock seconds per run.
RESULTS = {
    "Qwen2.5-0.5B\n(942 MB)": {
        "size_mb": 942.3,
        "store_s":  [],   # filled at runtime via argv / inline
        "gather_s": [],
    },
    "LFM2.5-350M\n(676 MB)": {
        "size_mb": 676.0,
        "store_s":  [],
        "gather_s": [],
    },
}

import sys, ast
if len(sys.argv) > 1:
    data = ast.literal_eval(sys.argv[1])
    for k, v in data.items():
        if k in RESULTS:
            RESULTS[k]["store_s"]  = v["store_s"]
            RESULTS[k]["gather_s"] = v["gather_s"]

# ── Style ─────────────────────────────────────────────────────────────────────
STORE_COLOR  = "#4C9BE8"
GATHER_COLOR = "#6FCF97"
ERR_COLOR    = "#555"
FONT = {"fontsize": 11}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def _bar_group(ax, models, store_means, store_errs, gather_means, gather_errs,
               ylabel, title, fmt="{:.0f}"):
    x = np.arange(len(models))
    w = 0.35
    b1 = ax.bar(x - w/2, store_means,  w, yerr=store_errs,  label="Store",
                color=STORE_COLOR,  capsize=5, error_kw={"ecolor": ERR_COLOR, "lw": 1.5})
    b2 = ax.bar(x + w/2, gather_means, w, yerr=gather_errs, label="Gather",
                color=GATHER_COLOR, capsize=5, error_kw={"ecolor": ERR_COLOR, "lw": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(models, **FONT)
    ax.set_ylabel(ylabel, **FONT)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + max(store_errs + gather_errs) * 0.05,
                    fmt.format(h), ha="center", va="bottom", fontsize=9, color="#333")


def compute_stats(vals):
    a = np.array(vals, dtype=float)
    return float(np.mean(a)), float(np.std(a)) if len(a) > 1 else 0.0


# ── Chart 1: Wall-clock time ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
models      = list(RESULTS.keys())
store_m, store_e   = zip(*[compute_stats(RESULTS[m]["store_s"])  for m in models])
gather_m, gather_e = zip(*[compute_stats(RESULTS[m]["gather_s"]) for m in models])

_bar_group(ax, models, store_m, list(store_e), gather_m, list(gather_e),
           ylabel="Wall-clock time (s)",
           title="Store & Gather Wall-Clock Time\n4 workers · 2× replication · ~100 Mbps Ethernet",
           fmt="{:.0f}s")
ax.set_ylim(0, max(max(store_m), max(gather_m)) * 1.3)
fig.tight_layout()
fig.savefig(OUT / "benchmark_wall_time.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}/benchmark_wall_time.png")


# ── Chart 2: Throughput ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
store_tp  = [RESULTS[m]["size_mb"] / RESULTS[m]["store_s"][i]  if RESULTS[m]["store_s"]  else 0
             for m in models for i in range(len(RESULTS[m]["store_s"]))]

def tp_stats(m, op):
    vals = [RESULTS[m]["size_mb"] / s for s in RESULTS[m][op]]
    return compute_stats(vals) if vals else (0.0, 0.0)

store_tm,  store_te  = zip(*[tp_stats(m, "store_s")  for m in models])
gather_tm, gather_te = zip(*[tp_stats(m, "gather_s") for m in models])

_bar_group(ax, models, store_tm, list(store_te), gather_tm, list(gather_te),
           ylabel="Aggregate throughput (MB/s)",
           title="Aggregate Throughput\n(checkpoint size ÷ wall-clock time)",
           fmt="{:.1f}")
ax.axhline(12.5, color="#E07B39", linestyle="--", linewidth=1.2, alpha=0.7, label="100 Mbps theoretical max")
ax.legend(fontsize=10)
ax.set_ylim(0, 14)
fig.tight_layout()
fig.savefig(OUT / "benchmark_throughput.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}/benchmark_throughput.png")


# ── Chart 3: Speedup vs old sequential ───────────────────────────────────────
# Gather speedup: measured via scripts/seq_baseline.py (sequential for-loop from
# git commit e4e04fc) vs parallel ThreadPoolExecutor gather wall times.
#   Qwen 942MB: 122.4s seq → 115.1s parallel  = 1.06×
#   LFM  676MB:  81.7s seq →  75.2s parallel  = 1.09×
# Store speedup: theoretical — old store did 2 sequential rounds of sends
# (primary then replica); new code does one parallel round → ~2× theoretical.
# No old sequential store binary to run: old /store-shard was HTTP-upload based,
# not the socket-push architecture, so direct comparison isn't meaningful.
SPEEDUP = {
    "Store\nQwen 942MB":  2.0,   # theoretical: 2 sequential rounds → 1 parallel
    "Gather\nQwen 942MB": round(122.4 / 115.1, 2),  # measured: seq_baseline.py
    "Store\nLFM 676MB":   2.0,
    "Gather\nLFM 676MB":  round(81.7 / 75.2, 2),   # measured: seq_baseline.py
}

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(list(SPEEDUP.keys()), list(SPEEDUP.values()),
              color=[STORE_COLOR, GATHER_COLOR, STORE_COLOR, GATHER_COLOR],
              width=0.5, zorder=3)
ax.axhline(1.0, color="#aaa", linestyle="--", linewidth=1)
ax.set_ylabel("Speedup vs. sequential baseline", **FONT)
ax.set_title("Parallel Speedup\n(ThreadPoolExecutor gather vs. sequential for-loop; store theoretical)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_ylim(0, 3.0)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
            f"{h:.1f}×", ha="center", va="bottom", fontsize=10, color="#333", fontweight="bold")
store_patch  = mpatches.Patch(color=STORE_COLOR,  label="Store")
gather_patch = mpatches.Patch(color=GATHER_COLOR, label="Gather")
ax.legend(handles=[store_patch, gather_patch], fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "benchmark_speedup.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}/benchmark_speedup.png")
