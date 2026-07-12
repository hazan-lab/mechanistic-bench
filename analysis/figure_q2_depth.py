"""Figure: compositional-depth correlation (Q2).

Across-architecture per-arch accuracy at hop depth k=2 predicts accuracy
at k=3 and k=4. This is the empirical handle on "completeness via depth":
if rankings at low depth predict rankings at high depth, the screen at
shallow tasks is informative about deeper compositions.

Also: theoretical anchor noted in caption to RASP/Tracr / log-prec TC0.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent
FIG_OUT = Path("/home/tt6444/mechanistic-bench/icml_workshop/figures/q2_compositional_depth.png")
FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

WIDE = pd.read_csv(ROOT / "unified_1m.csv").set_index("task")
keep_archs = ['attn', 'alt_attn_mamba', 'alt_attn_mamba2', 'alt_attn_stu', 'stu',
              'headwise', 'headwise_mamba2', 'headwise_stu', 'mamba']
W = WIDE[keep_archs].dropna()

# graph-traversal hop tasks -- same primitive at varying compositional depth
DEPTH = [
    ("two_hop", 2),
    ("three_hop", 3),
    ("deep_hop", 4),
]

ARCH_LABELS = {
    "attn": "Attn", "stu": "STU", "mamba": "Mamba",
    "alt_attn_mamba": "alt-Attn/Mamba", "alt_attn_mamba2": "alt-Attn/Mamba2",
    "alt_attn_stu": "alt-Attn/STU",
    "headwise": "head-Attn/Mamba", "headwise_mamba2": "head-Attn/Mamba2",
    "headwise_stu": "head-Attn/STU",
}
COLORS = {
    "attn":             "#1f77b4",
    "stu":              "#9467bd",
    "mamba":            "#d62728",
    "alt_attn_mamba":   "#2ca02c",
    "alt_attn_mamba2":  "#17becf",
    "alt_attn_stu":     "#8c564b",
    "headwise":         "#ff7f0e",
    "headwise_mamba2":  "#e377c2",
    "headwise_stu":     "#bcbd22",
}

fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6),
                          gridspec_kw={"wspace": 0.32})

# Panel a: per-arch accuracy curve as a function of depth
ax = axes[0]
for arch in keep_archs:
    ys = [W.loc[t, arch] for (t, _) in DEPTH]
    ax.plot([k for (_, k) in DEPTH], ys, "-o",
            label=ARCH_LABELS.get(arch, arch),
            color=COLORS[arch], alpha=0.85, lw=1.6, markersize=5)
ax.set_xticks([k for (_, k) in DEPTH])
ax.set_xlabel("Compositional depth k (hop-graph traversal)")
ax.set_ylabel("Token accuracy at 1m")
ax.set_title("(a) Per-arch accuracy at hop-depth k\n"
             "(harder with depth; rankings vary)", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="upper left", framealpha=0.92)

# Panel b: scatter k=2 vs k=4 -- two_hop values are clustered within
# ~0.02pt range, so per-point text annotations always collide. Switch
# to a legend keyed by color so the points keep their identity without
# any overlapping labels.
ax = axes[1]
xs = W.loc["two_hop"].values
ys = W.loc["deep_hop"].values
cs = [COLORS[a] for a in keep_archs]
ax.scatter(xs, ys, s=110, c=cs, edgecolors="k", linewidth=1, zorder=3)
z = np.polyfit(xs, ys, 1)
xx = np.linspace(min(xs) - 0.002, max(xs) + 0.002, 50)
ax.plot(xx, np.polyval(z, xx), "--", c="gray", alpha=0.7, zorder=1)
rho, prho = spearmanr(xs, ys)
r, pr = pearsonr(xs, ys)
ax.set_xlabel("two_hop (k=2) accuracy")
ax.set_ylabel("deep_hop (k=4) accuracy")
ax.set_title(f"(b) k=2 predicts k=4\n"
             f"Spearman $\\rho$={rho:.2f} (p={prho:.3f}),  "
             f"Pearson $r$={r:.2f}", fontsize=10)
ax.grid(alpha=0.3)
xpad = (xs.max() - xs.min()) * 0.10
ypad = (ys.max() - ys.min()) * 0.08
ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
ax.set_ylim(ys.min() - ypad, ys.max() + ypad * 1.4)
legend_handles = [
    plt.Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=COLORS[a], markeredgecolor="k",
               markersize=8, label=ARCH_LABELS[a])
    for a in keep_archs
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
          frameon=True, framealpha=0.92, handletextpad=0.4,
          borderpad=0.35, labelspacing=0.25)

# Panel c: heatmap of pairwise Spearman across hop-depth tasks
ax = axes[2]
TASKS = ["two_hop", "three_hop", "deep_hop", "k_hop"]
M = np.zeros((len(TASKS), len(TASKS)))
for i, ti in enumerate(TASKS):
    for j, tj in enumerate(TASKS):
        rho, _ = spearmanr(W.loc[ti], W.loc[tj])
        M[i, j] = rho
im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(TASKS)))
ax.set_yticks(range(len(TASKS)))
ax.set_xticklabels(TASKS, rotation=30, ha="right")
ax.set_yticklabels(TASKS)
for i in range(len(TASKS)):
    for j in range(len(TASKS)):
        ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                color="white" if abs(M[i,j]) > 0.5 else "black", fontsize=9)
ax.set_title("(c) Across-arch Spearman ρ\nbetween hop-depth tasks",
             fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.04)

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
print(f"wrote {FIG_OUT}")
print(f"two_hop -> deep_hop: ρ={spearmanr(W.loc['two_hop'], W.loc['deep_hop'])}")
