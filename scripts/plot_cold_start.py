"""Regenerate cold_start_barchart.pdf for paper §5.2.

Left panel: rank_median per bucket (lower = better).
Right panel: Recall@1000 per bucket (higher = better).
Data sourced from code/benchmark/results/analysis/cold_start_5seeds.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _paths import code  # dual-layout paths: dev tree or published release

ROOT = Path(__file__).resolve().parent.parent
SRC = code("benchmark", "results", "analysis", "cold_start_5seeds.json")
OUT_PAPER = ROOT / "paper" / "cold_start_barchart.pdf"
OUT_DOCS_PNG = ROOT / "docs" / "figures" / "cold_start_barchart.png"

with open(SRC) as f:
    data = json.load(f)

CONFIGS = ["M1", "M4", "M7"]
LABELS = {
    "M1": "M1: LightGCN (ID only)",
    "M4": "M4: + LLM Profile",
    "M7": "M7: + Profile + Mood",
}
COLORS = {"M1": "#9e9e9e", "M4": "#1f77b4", "M7": "#2ca02c"}
BUCKETS = ["cold", "medium", "warm"]
BUCKET_LABELS = ["Cold (<10)", "Medium (10-49)", "Warm (\u226550)"]


def extract(metric):
    out = {cfg: [] for cfg in CONFIGS}
    for cfg in CONFIGS:
        for b in BUCKETS:
            out[cfg].append(data["configs"][cfg]["buckets"][b][metric]["mean"])
    return out


rank_med = extract("rank_median")
recall_1000 = extract("Recall@1000")

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: rank_median (log scale — cold vs warm span 2 orders of magnitude)
x = np.arange(len(BUCKETS))
width = 0.26
for i, cfg in enumerate(CONFIGS):
    offset = (i - 1) * width
    bars = ax_l.bar(x + offset, rank_med[cfg], width,
                    label=LABELS[cfg], color=COLORS[cfg],
                    edgecolor="white", linewidth=0.5)
    for rect, v in zip(bars, rank_med[cfg]):
        ax_l.text(rect.get_x() + rect.get_width()/2,
                  v * 1.08, f"{int(v):,}",
                  ha="center", va="bottom", fontsize=8)

ax_l.set_yscale("log")
ax_l.set_xticks(x)
ax_l.set_xticklabels(BUCKET_LABELS, fontsize=10)
ax_l.set_ylabel("Rank_median  (lower is better)")
ax_l.set_title("Median rank of held-out GT items", fontsize=11)
ax_l.set_ylim(top=max(max(v) for v in rank_med.values()) * 2.3)
ax_l.grid(axis="y", linestyle="--", alpha=0.4, which="both")
ax_l.set_axisbelow(True)
ax_l.legend(loc="upper right", fontsize=8, framealpha=0.9)

# Right: Recall@1000
for i, cfg in enumerate(CONFIGS):
    offset = (i - 1) * width
    bars = ax_r.bar(x + offset, recall_1000[cfg], width,
                    label=LABELS[cfg], color=COLORS[cfg],
                    edgecolor="white", linewidth=0.5)
    for rect, v in zip(bars, recall_1000[cfg]):
        if v >= 0.001:
            txt = f"{v:.3f}" if v < 0.1 else f"{v:.2f}"
        else:
            txt = "0"
        ax_r.text(rect.get_x() + rect.get_width()/2,
                  v + max(recall_1000["M1"]) * 0.015, txt,
                  ha="center", va="bottom", fontsize=8)

ax_r.set_xticks(x)
ax_r.set_xticklabels(BUCKET_LABELS, fontsize=10)
ax_r.set_ylabel("Recall@1000  (higher is better)")
ax_r.set_title("Recall@1000 (top 10\\% of 9,906-item catalog)", fontsize=11)
ax_r.set_ylim(0, max(max(v) for v in recall_1000.values()) * 1.15)
ax_r.grid(axis="y", linestyle="--", alpha=0.4)
ax_r.set_axisbelow(True)
ax_r.legend(loc="upper left", fontsize=8, framealpha=0.9)

plt.tight_layout()

OUT_DOCS_PNG.parent.mkdir(parents=True, exist_ok=True)
if OUT_PAPER.parent.exists(): fig.savefig(OUT_PAPER, bbox_inches="tight")
fig.savefig(OUT_DOCS_PNG, bbox_inches="tight", dpi=200)

print(f"rank_median:   {rank_med}")
print(f"Recall@1000:   {recall_1000}")
print(f"Wrote {OUT_PAPER}")
print(f"Wrote {OUT_DOCS_PNG}")
