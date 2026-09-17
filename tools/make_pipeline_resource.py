"""Draw Figure 1, the three-stage pipeline diagram, as a vector PDF.

Regenerates figures/pipeline_resource.pdf from the layout in this file, so
the figure in the paper and the one in the release come from one source.
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13.5, 3.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# stage boxes
B_TOP, B_BOT = 95, 38
stages = [
    dict(x=1.0, w=31, color="#dceaf7", edge="#2f6db0", title="Stage 1 — Profile Generation",
         inp="ML-20M genome tags\n+ TMDb metadata",
         model="Claude Haiku 4.5",
         out="10,381 profiles\n80–120-word text +\n10-axis mood + themes"),
    dict(x=34.5, w=31, color="#dcf0e3", edge="#2e8b57", title="Stage 2 — Embedding",
         inp="Profile text /\nmood / themes",
         model="bge-large-en-v1.5",
         out="1024-d profile +\n10-d mood +\n528-d themes"),
    dict(x=68.0, w=31, color="#f9dede", edge="#c0392b", title="Stage 3 — Evaluation",
         inp="M1 pure CF · M2 genome-PCA\nM3 title-emb · M4 profile\nM7 profile+mood",
         model="Full-ranking · 5 seeds",
         out="NDCG@K, Recall@K, MRR\ntemporal split · full ranking\ncold / medium / warm buckets"),
]

mid_y = (B_TOP + B_BOT) / 2
for s in stages:
    ax.add_patch(FancyBboxPatch((s["x"], B_BOT), s["w"], B_TOP - B_BOT,
                 boxstyle="round,pad=0.4,rounding_size=2.0",
                 linewidth=1.6, edgecolor=s["edge"], facecolor=s["color"]))
    cx = s["x"] + s["w"] / 2
    ax.text(cx, 89.5, s["title"], ha="center", va="center",
            fontsize=14, fontweight="bold", color=s["edge"])
    ax.text(cx, 79.0, s["inp"], ha="center", va="center",
            fontsize=11, color="#1a1a1a", linespacing=1.25)
    ax.text(cx, 66.0, "▼  " + s["model"], ha="center", va="center",
            fontsize=12, fontweight="bold", color=s["edge"])
    ax.text(cx, 50.0, s["out"], ha="center", va="center",
            fontsize=11, color="#1a1a1a", linespacing=1.25)

# inter-stage arrows
for x0, x1 in [(32.0, 34.5), (65.5, 68.0)]:
    ax.add_patch(FancyArrowPatch((x0, mid_y), (x1, mid_y),
                 arrowstyle="-|>", mutation_scale=22, linewidth=2.0, color="#444"))

# terminal framework banner
ax.add_patch(FancyBboxPatch((1.0, 12.0), 98.0, 18.0,
             boxstyle="round,pad=0.4,rounding_size=2.0",
             linewidth=1.6, edgecolor="#6c3483", facecolor="#efe3f5"))
ax.text(50, 21.0,
        "Construct validity: the profile beats the genome vector, the same encoder on titles, and pure CF —\n"
        "and survives removal of every labelled popularity input from the prompt  (§5, §6)",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color="#6c3483", linespacing=1.25)
ax.add_patch(FancyArrowPatch((83.5, B_BOT), (83.5, 30.0),
             arrowstyle="-|>", mutation_scale=20, linewidth=2.0, color="#6c3483"))

ax.text(50, 5.5, "Stages 1–2 are one-time, offline: ≈ \\$21 one-time; prompt caching cannot engage.",
        ha="center", va="center", fontsize=10, style="italic", color="#555")

plt.tight_layout(pad=0.3)
# Resolve relative to this file, not the caller's cwd: the figure gate and the
# Makefile invoke this from different directories.
OUT = pathlib.Path(__file__).resolve().parent.parent / "figures" / "pipeline_resource.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
