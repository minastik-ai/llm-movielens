"""Draw Figure 1, the three-stage pipeline diagram, as a vector PDF.

Regenerates figures/pipeline_resource.pdf from the layout in this file, so
the figure in the paper and the one in the release come from one source.

LEGIBILITY IS A LAYOUT CONSTRAINT, NOT A TASTE QUESTION, so this file asserts
it. The figure is placed at `width=\\textwidth`, so every font size in it is
multiplied by TEXTWIDTH_PT / (FIG_W * 72) before a reader sees it. At the
original 13.5in width that factor was 0.354 and the whole diagram -- 702
characters -- printed under 4pt. Nothing in the source says "11" and means
"3.5", which is exactly why this went unnoticed through four audits:
matplotlib reports the number you typed.

Two invariants follow, both checked below against the real renderer rather
than estimated, so the figure cannot regress silently:

  1. every string renders at >= MIN_RENDERED_PT on the page, and
  2. every string fits inside the box that contains it.

Note that the rendered HEIGHT of the figure depends only on the ASPECT ratio
(h/w * TEXTWIDTH_PT), never on the absolute inches -- so shrinking the canvas
buys legibility at a fixed page footprint, up to the point where the text no
longer fits. That is the trade this layout is tuned against.
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# llncs \textwidth, measured with \the\textwidth, not assumed.
TEXTWIDTH_PT = 347.12
MIN_RENDERED_PT = 6.0          # the paper's own footnote type is 6.97pt
MARGIN_PX = 4                  # containment alone passes text touching the rule
FIG_W, FIG_H = 8.6, 2.4
SCALE = TEXTWIDTH_PT / (FIG_W * 72)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

B_TOP, B_BOT = 98, 30
stages = [
    dict(x=1.0, w=31, color="#dceaf7", edge="#2f6db0", title="Stage 1 — Generation",
         inp="ML-20M genome tags\n+ TMDb metadata",
         model="Claude Haiku 4.5",
         out="10,381 profiles\n80–120-word text +\n10-axis mood + themes"),
    dict(x=34.5, w=31, color="#dcf0e3", edge="#2e8b57", title="Stage 2 — Embedding",
         inp="Profile text /\nmood / themes",
         model="bge-large-en-v1.5",
         out="1024-d profile +\n10-d mood +\n528-d themes"),
    # "temporal split" left out on purpose: the model row already says
    # "Full-ranking", and §4 states the split. A figure that repeats the
    # protocol costs the vertical space legibility needs.
    # "M2 genome" not "M2 genome-PCA": the longer form is 2px too wide for the
    # box, and the label M2 already distinguishes it from M2b, the raw genome.
    dict(x=68.0, w=31, color="#f9dede", edge="#c0392b", title="Stage 3 — Evaluation",
         inp="M1 CF · M2 genome · M3 title\nM4 profile · M7 profile+mood",
         model="Full-ranking · 5 seeds",
         out="NDCG@K, Recall@K, MRR\ncold/medium/warm buckets"),
]

FS_TITLE, FS_MODEL, FS_BODY, FS_BANNER = 13, 12, 11, 11
mid_y = (B_TOP + B_BOT) / 2
placed = []                     # (text artist, containing patch) for the fit check

for s in stages:
    box = FancyBboxPatch((s["x"], B_BOT), s["w"], B_TOP - B_BOT,
                         boxstyle="round,pad=0.4,rounding_size=2.0",
                         linewidth=1.6, edgecolor=s["edge"], facecolor=s["color"])
    ax.add_patch(box)
    cx = s["x"] + s["w"] / 2
    for y, txt, fs, bold in [(91.7, s["title"], FS_TITLE, True),
                             (77.5, s["inp"], FS_BODY, False),
                             (63.6, "▼  " + s["model"], FS_MODEL, True),
                             (45.7, s["out"], FS_BODY, False)]:
        placed.append((ax.text(cx, y, txt, ha="center", va="center", fontsize=fs,
                               fontweight="bold" if bold else "normal",
                               color=s["edge"] if bold else "#1a1a1a",
                               linespacing=1.25), box))

for x0, x1 in [(32.0, 34.5), (65.5, 68.0)]:
    ax.add_patch(FancyArrowPatch((x0, mid_y), (x1, mid_y),
                 arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#444"))

banner = FancyBboxPatch((1.0, 3.0), 98.0, 21.0,
                        boxstyle="round,pad=0.4,rounding_size=2.0",
                        linewidth=1.6, edgecolor="#6c3483", facecolor="#efe3f5")
ax.add_patch(banner)
placed.append((ax.text(
    50, 13.5,
    "Construct validity: the profile beats the genome vector, the same encoder on\n"
    "titles, and pure CF — and survives removing every popularity input  (§5, §6)",
    ha="center", va="center", fontsize=FS_BANNER, fontweight="bold",
    color="#6c3483", linespacing=1.25), banner))
ax.add_patch(FancyArrowPatch((83.5, B_BOT), (83.5, 24.0),
             arrowstyle="-|>", mutation_scale=17, linewidth=1.8, color="#6c3483"))

# The one-time cost used to sit here as a hand-typed "= $21". It is \ecCostExact
# ($20.66) in §8 with the caching explanation beside it; a second, rounded,
# ungenerated copy inside a PDF is a number no gate can reach (Invariant 23).

plt.tight_layout(pad=0.3)

# --- the two invariants, checked against the real renderer -------------------
fig.canvas.draw()
r = fig.canvas.get_renderer()
fails = []
for t, box in placed:
    pt = t.get_fontsize() * SCALE
    if pt < MIN_RENDERED_PT:
        fails.append(f"{t.get_text()[:34]!r} renders at {pt:.2f}pt on the page "
                     f"(floor {MIN_RENDERED_PT}pt)")
    tb = t.get_window_extent(r)
    bb = box.get_window_extent(r)
    m = MARGIN_PX
    if not (tb.x0 >= bb.x0 + m and tb.x1 <= bb.x1 - m
            and tb.y0 >= bb.y0 + m and tb.y1 <= bb.y1 - m):
        over = {"left": bb.x0 + m - tb.x0, "right": tb.x1 - (bb.x1 - m),
                "bottom": bb.y0 + m - tb.y0, "top": tb.y1 - (bb.y1 - m)}
        worst = ", ".join(f"{k} {v:+.0f}px" for k, v in over.items() if v > 0)
        fails.append(f"{t.get_text()[:34]!r} overflows its box: {worst}")
for i, (t1, b1) in enumerate(placed):
    for t2, b2 in placed[i + 1:]:
        if b1 is not b2: continue
        e1, e2 = t1.get_window_extent(r), t2.get_window_extent(r)
        if e1.overlaps(e2):
            fails.append(f"{t1.get_text()[:26]!r} and {t2.get_text()[:26]!r} "
                         f"collide inside the same box")

if fails:
    raise SystemExit("figure 1 layout FAILED:\n  " + "\n  ".join(fails))

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures" / "pipeline_resource.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}: {len(placed)} strings, smallest renders at "
      f"{min(t.get_fontsize() for t, _ in placed) * SCALE:.2f}pt at "
      f"\\textwidth={TEXTWIDTH_PT}pt; figure occupies "
      f"{TEXTWIDTH_PT * FIG_H / FIG_W:.0f}pt of column height")
