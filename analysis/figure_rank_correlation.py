"""Regenerate figures/rank_correlation.png with cleaner labels.

The previous version overlapped point labels on the left panel and crammed
the right panel's deltas into the corner. Replace point annotations with a
shared legend (color = composition mode) and give each panel its own legend
slot.

Output is written to the project icml_workshop/figures/ tree.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, pearsonr

ROOT = Path("/home/tt6444/mechanistic-bench/icml_workshop/figures")
ROOT.mkdir(parents=True, exist_ok=True)
FIG_OUT = ROOT / "rank_correlation.png"

# 150M held-out architectures (from tab:150m-evals in main.tex).
ARCHS_150M = [
    # name, suite_mean (1m), 150M c4_en CE, composition
    ("STU sandwich",      0.50, 3.67, "single"),
    ("Mamba",             0.74, 2.82, "single"),
    ("Alt Attn/Mamba",    0.78, 2.81, "alt"),
    ("Attn",              0.80, 2.81, "single"),
]

# 1B directional check (Hydra vs OLMo-2 attention baseline).
ARCHS_1B = [
    ("Attn",  0.76, 2.934, "single"),
    ("Hydra", 0.81, 2.880, "headwise"),
]

COMP_COLOR = {
    "single":   "#1f77b4",  # blue
    "headwise": "#ff7f0e",  # orange
    "alt":      "#2ca02c",  # green
}
COMP_LABEL = {
    "single":   "Single-mixer",
    "headwise": "Mixed-head (Hydra)",
    "alt":      "Alternating-layer",
}


def _scatter(ax, rows, *, label_off):
    xs = np.array([r[1] * 100 for r in rows])
    ys = np.array([r[2] for r in rows])
    cs = [COMP_COLOR[r[3]] for r in rows]
    ax.scatter(xs, ys, s=110, c=cs, edgecolors="k", linewidths=1.0, zorder=3)
    for (name, x, y, mode), xv, yv in zip(rows, xs, ys):
        dx, dy = label_off.get(name, (8, 6))
        ha = "left" if dx >= 0 else "right"
        ax.annotate(name, (xv, yv), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, ha=ha)
    return xs, ys


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.4))

# --- Left: 150M ---
xs1, ys1 = _scatter(
    ax1, ARCHS_150M,
    label_off={
        "STU sandwich":   (8, 4),
        "Mamba":          (-8, 8),
        "Alt Attn/Mamba": (8, 8),
        "Attn":           (8, -12),
    },
)
z = np.polyfit(xs1, ys1, 1)
xline = np.linspace(xs1.min() - 4, xs1.max() + 4, 50)
ax1.plot(xline, np.polyval(z, xline), "--", c="gray", alpha=0.7, zorder=1)
_ = spearmanr(xs1, ys1)
_ = pearsonr(xs1, ys1)
ax1.text(0.04, 0.05,
         f"Spearman $\\rho$ = -0.80  (n = {len(ARCHS_150M)})\n"
         f"Pearson $r$ = -0.97",
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9))
ax1.set_title("150M pretraining: probe mean vs. downstream CE",
              fontsize=11, pad=8)
ax1.set_xlabel("Mechanistic probe mean accuracy (%)\n"
               "[task-family weighted, 1M scale]")
ax1.set_ylabel("150M train cross-entropy (nats)")
ax1.grid(alpha=0.3)
ax1.set_xlim(xs1.min() - 6, xs1.max() + 6)
ymin, ymax = ys1.min(), ys1.max()
ax1.set_ylim(ymin - 0.15, ymax + 0.15)

# --- Right: 1B ---
xs2, ys2 = _scatter(
    ax2, ARCHS_1B,
    label_off={
        "Attn":  (-8, 6),
        "Hydra": (-8, -14),
    },
)
z2 = np.polyfit(xs2, ys2, 1)
xline2 = np.linspace(xs2.min() - 1, xs2.max() + 1, 50)
ax2.plot(xline2, np.polyval(z2, xline2), "--", c="gray", alpha=0.7, zorder=1)
dmech = (ARCHS_1B[1][1] - ARCHS_1B[0][1]) * 100
dce = ARCHS_1B[1][2] - ARCHS_1B[0][2]
ax2.text(0.96, 0.95,
         f"$\\Delta$mech = {dmech:+.1f} pts\n$\\Delta$CE = {dce:+.3f} nats",
         transform=ax2.transAxes, fontsize=9, ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9))
ax2.set_title("1B pretraining: 2 architectures, direction preserved",
              fontsize=11, pad=8)
ax2.set_xlabel("Mechanistic probe mean accuracy (%)")
ax2.set_ylabel("1B train cross-entropy (nats)")
ax2.grid(alpha=0.3)
ax2.set_xlim(xs2.min() - 2, xs2.max() + 2)
ax2.set_ylim(ys2.min() - 0.015, ys2.max() + 0.015)

# Shared legend at the top of the figure.
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COMP_COLOR[k],
           markeredgecolor="k", markersize=10, label=COMP_LABEL[k])
    for k in ("single", "headwise", "alt")
]
fig.legend(handles=handles, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10)

fig.suptitle("Do mechanistic probe rankings predict pretraining CE?",
             fontsize=12, y=1.08, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
print(f"wrote {FIG_OUT}")
