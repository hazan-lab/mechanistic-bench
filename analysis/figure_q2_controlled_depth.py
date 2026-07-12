"""Q2 figure: controlled compositional-depth sweep on deep_hop, k=1..5,
across 5 architectures.

Layout fix: in panel (b) the four high-accuracy points (Attn, Mamba-2,
alt-Attn/Mamba, head-Attn/Mamba) cluster together so direct annotations
collide. Replace per-point text with an in-panel legend keyed by color and
keep direct labels only on the well-separated outliers (STU); use leader
arrows for the cluster so labels never overlap each other.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent
FIG_OUT = Path("/home/tt6444/mechanistic-bench/icml_workshop/figures/q2_controlled.png")
FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "depth_sweep.csv")
archs = sorted(df["arch"].unique())
ks = sorted(df["k"].unique())
mat = df.pivot(index="arch", columns="k", values="tok_acc")

ARCH_LABELS = {
    "attn": "Attn", "stu": "STU", "mamba": "Mamba", "mamba2": "Mamba-2",
    "alt_attn_mamba": "alt-Attn/Mamba", "headwise": "head-Attn/Mamba",
}
COLORS = {
    "attn": "#1f77b4", "stu": "#9467bd", "mamba": "#d62728",
    "mamba2": "#e377c2", "alt_attn_mamba": "#2ca02c", "headwise": "#ff7f0e",
}

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6),
                          gridspec_kw={"wspace": 0.32})

# -------- Panel (a): per-arch depth curves --------
ax = axes[0]
for a in archs:
    if a in mat.index:
        ys = mat.loc[a].dropna()
        ax.plot(ys.index, ys.values, "-o", label=ARCH_LABELS.get(a, a),
                color=COLORS.get(a, "gray"), lw=2, markersize=7)
ax.set_xticks(ks)
ax.set_xlabel("Compositional depth k (deep_hop)")
ax.set_ylabel("Token accuracy at 1m, max_steps=2000")
ax.set_title("(a) Controlled depth sweep on deep_hop\n"
             "(same graph generator, k varies)", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="upper right", framealpha=0.92)

# -------- Panel (b): k=4 vs k=5 scatter --------
ax = axes[1]
common = mat[[4, 5]].dropna()
xs = common[4].values.astype(float)
ys = common[5].values.astype(float)
names = list(common.index)

# Compute fit on the non-outlier set (everyone except STU at k=5).
ax.scatter(xs, ys, s=110,
           c=[COLORS.get(a, "gray") for a in names],
           edgecolors="k", linewidth=1, zorder=3,
           label=None)

# Per-arch legend for panel (b) so labels do not have to sit next to points.
legend_handles = [
    plt.Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=COLORS.get(a, "gray"),
               markeredgecolor="k", markersize=9,
               label=ARCH_LABELS.get(a, a))
    for a in names
]

z = np.polyfit(xs, ys, 1)
xx = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 50)
ax.plot(xx, np.polyval(z, xx), "--", c="gray", alpha=0.7, zorder=1)

rho_res = spearmanr(xs, ys)
rho = rho_res.statistic if hasattr(rho_res, "statistic") else rho_res[0]
prho = rho_res.pvalue if hasattr(rho_res, "pvalue") else rho_res[1]
r_res = pearsonr(xs, ys)
r = r_res.statistic if hasattr(r_res, "statistic") else r_res[0]
pr = r_res.pvalue if hasattr(r_res, "pvalue") else r_res[1]
ax.set_title("(b) k=4 predicts k=5 (controlled)\n"
             f"$\\rho$={rho:.2f} (p={prho:.3f}),  r={r:.2f} (p={pr:.3f})",
             fontsize=10)
ax.set_xlabel("deep_hop k=4 token accuracy")
ax.set_ylabel("deep_hop k=5 token accuracy")
ax.grid(alpha=0.3)

# Slightly expand limits to leave room for the legend in the lower-right.
xpad = (xs.max() - xs.min()) * 0.10
ypad = (ys.max() - ys.min()) * 0.08
ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
ax.set_ylim(ys.min() - ypad, ys.max() + ypad * 1.6)

ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
          frameon=True, framealpha=0.92, handletextpad=0.4)

# -------- Panel (c): pairwise across-arch Spearman ρ --------
ax = axes[2]
M = np.full((len(ks), len(ks)), np.nan)
for i, ki in enumerate(ks):
    for j, kj in enumerate(ks):
        if ki in mat.columns and kj in mat.columns:
            common = mat[[ki, kj]].dropna()
            if len(common) > 2:
                rho_res = spearmanr(common[ki], common[kj])
                rho_val = rho_res.statistic if hasattr(rho_res, "statistic") else rho_res[0]
                if isinstance(rho_val, np.ndarray):
                    rho_val = float(rho_val.flat[0]) if rho_val.size == 1 else np.nan
                M[i, j] = float(rho_val)
im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(ks)))
ax.set_yticks(range(len(ks)))
ax.set_xticklabels([f"k={k}" for k in ks])
ax.set_yticklabels([f"k={k}" for k in ks])
for i in range(len(ks)):
    for j in range(len(ks)):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    color="white" if abs(M[i, j]) > 0.5 else "black",
                    fontsize=10)
ax.set_title(f"(c) Across-arch Spearman $\\rho$\n"
             f"between depths (n={mat.shape[0]} archs)", fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
print(f"wrote {FIG_OUT}")
