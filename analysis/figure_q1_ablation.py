"""Figure: drop-primitive ablation (Q1).

Two-panel figure:
  (a) Scatter: full-suite mean accuracy vs 50M c4_en CE for n=9 archs.
  (b) Bar: Spearman ρ after dropping each primitive.

Layout fix: previously the external legend on (b) was anchored to the right
side, which pushed the bar chart leftward and made its long y-tick labels
collide with (a)'s x-axis labels. Now the per-color legend lives inside (b)
along the bottom-right, and the two panels get more breathing room via a
larger wspace and an explicit subplots_adjust.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr
from primitive_taxonomy import PRIMARY_PRIMITIVE  # noqa: F401

ROOT = Path(__file__).resolve().parent
FIG_OUT = Path("/home/tt6444/mechanistic-bench/icml_workshop/figures/q1_drop_family_ablation.png")
FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

WIDE = pd.read_csv(ROOT / "unified_1m.csv").set_index("task")
LM = pd.read_csv(ROOT / "lm_ce.csv")
LM50 = LM[LM.scale == "50m"].set_index("arch")
abl = pd.read_csv(ROOT / "drop_family_ablation.csv")
abl_50 = abl[(abl.scale == "50m") & (abl.level == "primitive")].copy()
baseline_50 = abl[(abl.scale == "50m") & (abl.level == "baseline")].iloc[0]

keep_archs = ['attn', 'alt_attn_mamba', 'alt_attn_mamba2', 'alt_attn_stu', 'stu',
              'headwise', 'headwise_mamba2', 'headwise_stu', 'mamba']
WIDE = WIDE[keep_archs].dropna()
suite_mean = WIDE.mean(axis=0)
ce = LM50.loc[keep_archs, "c4_en_CE"]

ARCH_LABELS = {
    "attn": "Attn", "stu": "STU", "mamba": "Mamba",
    "alt_attn_mamba": "alt-Attn/Mamba", "alt_attn_mamba2": "alt-Attn/Mamba2",
    "alt_attn_stu": "alt-Attn/STU",
    "headwise": "head Attn+Mamba", "headwise_mamba2": "head Attn+Mamba2",
    "headwise_stu": "head Attn+STU",
}

# Custom annotation offsets to avoid overlap.
ANN_OFF = {
    "attn":             (-8, 8),
    "stu":              (8, 6),
    "mamba":            (-8, -4),
    "alt_attn_mamba":   (8, -10),
    "alt_attn_mamba2":  (8, 8),
    "alt_attn_stu":     (-8, -12),
    "headwise":         (8, 4),
    "headwise_mamba2":  (-10, -10),
    "headwise_stu":     (-8, 8),
}

fig = plt.figure(figsize=(13.0, 5.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.55)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ---------- Panel (a) ----------
ax1.scatter(suite_mean, ce, s=85, c="#1f77b4", edgecolors="k",
            linewidth=1, zorder=3)
for arch in keep_archs:
    dx, dy = ANN_OFF.get(arch, (6, 4))
    ha = "left" if dx >= 0 else "right"
    ax1.annotate(ARCH_LABELS.get(arch, arch),
                 (suite_mean[arch], ce[arch]),
                 textcoords="offset points", xytext=(dx, dy),
                 fontsize=8, ha=ha)
z = np.polyfit(suite_mean.values, ce.values, 1)
xs = np.linspace(suite_mean.min() * 0.97, suite_mean.max() * 1.02, 50)
ax1.plot(xs, np.polyval(z, xs), "--", c="gray", alpha=0.7, zorder=1)
rho, prho = spearmanr(suite_mean, ce)
r, pr = pearsonr(suite_mean, ce)
ax1.set_title(f"(a) Held-out validation @ 50M, n={len(keep_archs)} archs\n"
              f"Spearman $\\rho$={rho:.2f} (p={prho:.3f}),  "
              f"Pearson $r$={r:.2f} (p={pr:.3f})",
              fontsize=10)
ax1.set_xlabel("mech-suite mean accuracy @ 1m (62 tasks)")
ax1.set_ylabel("50M LM c4_en cross-entropy (lower=better)")
ax1.grid(alpha=0.3)
xpad = (suite_mean.max() - suite_mean.min()) * 0.10
ypad = (ce.max() - ce.min()) * 0.08
ax1.set_xlim(suite_mean.min() - xpad, suite_mean.max() + xpad)
ax1.set_ylim(ce.min() - ypad, ce.max() + ypad)

# ---------- Panel (b) ----------
abl_50 = abl_50.assign(abs_rho=abl_50["rho"].abs())
abl_50 = abl_50.sort_values("abs_rho", ascending=True)
y = np.arange(len(abl_50))
baseline_abs = abs(baseline_50["rho"])
colors = []
for _, row in abl_50.iterrows():
    delta = abs(row["rho"]) - baseline_abs
    if delta < -0.02:
        colors.append("#d62728")  # load-bearing (drop weakens)
    elif delta > 0.02:
        colors.append("#2ca02c")  # over-weighted (drop strengthens)
    else:
        colors.append("#888888")  # robust

ax2.barh(y, abl_50["rho"], color=colors, edgecolor="k", lw=0.5)
ax2.set_yticks(y)
ax2.set_yticklabels(abl_50["drop"], fontsize=9)
ax2.axvline(baseline_50["rho"], ls="--", c="k", lw=1.2)
ax2.set_xlabel("Spearman $\\rho$ after dropping primitive\n"
               "(more negative = stronger correlation)")
ax2.set_title(f"(b) Drop-primitive ablation @ 50M (n={int(baseline_50['n_archs'])} archs)",
              fontsize=10)
ax2.grid(axis="x", alpha=0.3)

# Bars are drawn from 0 to ρ (negative), so the only bar-free strip is to
# the right of x=0. Extend xlim there to host a clean p-value column that
# never crosses the dashed baseline line (which sits between -0.75 and -0.70).
xmin = float(abl_50["rho"].min())
ax2.set_xlim(xmin - 0.02, 0.18)

p_x = 0.01
for i, (_, row) in enumerate(abl_50.iterrows()):
    ax2.text(p_x, i, f"p={row['rho_p']:.3f}",
             va="center", ha="left", fontsize=8, color="black")

# Legend placed *above* panel (b) so it never covers the last bar.
handles = [
    Patch(color="#d62728", label="load-bearing"),
    Patch(color="#888888", label="robust ($\\approx$ baseline)"),
    Patch(color="#2ca02c", label="over-weighted"),
    plt.Line2D([0], [0], color="k", ls="--",
               label=f"baseline $\\rho$ = {baseline_50['rho']:.2f}"),
]
ax2.legend(handles=handles, loc="lower center",
           bbox_to_anchor=(0.5, 1.10),
           fontsize=8, frameon=False, ncol=4,
           handletextpad=0.4, columnspacing=1.2)

plt.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
print(f"wrote {FIG_OUT}")
