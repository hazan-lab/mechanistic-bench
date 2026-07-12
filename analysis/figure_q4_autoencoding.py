"""Figure: autoencoding/compression-flavored probes (Q4).

Shows that the three reconstruction-flavored probes already in the
suite (noisy_copy, compress, reverse_copy) at 1m predict 50M LM
wikitext_103 cross-entropy at Spearman ρ=-0.83 (p=0.005, n=9 archs),
on par with the full-suite correlation. The screening claim is
therefore not specific to causal next-token-prediction behavior.

Layout fix: hand-placed annotation offsets for the right-side cluster
(alt-Attn/{Mamba, Mamba2, STU}) so labels no longer overlap; in panel
(b), per-task ρ/p text moved to the right of the bars (outside the
fill) so it stops colliding with the y-tick labels.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent
FIG_OUT = Path("/home/tt6444/mechanistic-bench/icml_workshop/figures/q4_autoencoding.png")
FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

WIDE = pd.read_csv(ROOT / "unified_1m.csv").set_index("task")
LM = pd.read_csv(ROOT / "lm_ce.csv")
LM50 = LM[LM.scale == "50m"].set_index("arch")

keep_archs = ['attn', 'alt_attn_mamba', 'alt_attn_mamba2', 'alt_attn_stu', 'stu',
              'headwise', 'headwise_mamba2', 'headwise_stu', 'mamba']
WIDE = WIDE[keep_archs].dropna()

ARCH_LABELS = {
    "attn": "Attn", "stu": "STU", "mamba": "Mamba",
    "alt_attn_mamba": "alt-Attn/Mamba", "alt_attn_mamba2": "alt-Attn/Mamba2",
    "alt_attn_stu": "alt-Attn/STU",
    "headwise": "head-Attn/Mamba", "headwise_mamba2": "head-Attn/Mamba2",
    "headwise_stu": "head-Attn/STU",
}

ANN_OFF = {
    "attn":             (-8, 8),
    "stu":              (8, 4),
    "mamba":            (8, 4),
    "alt_attn_mamba":   (8, -10),
    "alt_attn_mamba2":  (8, 8),
    "alt_attn_stu":     (8, -10),
    "headwise":         (8, 8),
    "headwise_mamba2":  (8, -10),
    "headwise_stu":     (-8, 8),
}

AE_TASKS = ["noisy_copy", "compress", "reverse_copy"]
ae_mean = WIDE.loc[AE_TASKS].mean(axis=0)
target = LM50.loc[keep_archs, "wikitext_103_CE"]

fig = plt.figure(figsize=(12.5, 4.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.35)

# -------- Panel (a): AE-mean scatter --------
ax = fig.add_subplot(gs[0])
ax.scatter(ae_mean, target, s=85, c="#2ca02c", edgecolors="k",
           linewidth=1, zorder=3)
for arch in keep_archs:
    dx, dy = ANN_OFF.get(arch, (5, 4))
    ha = "left" if dx >= 0 else "right"
    ax.annotate(ARCH_LABELS.get(arch, arch),
                (ae_mean[arch], target[arch]),
                textcoords="offset points", xytext=(dx, dy),
                fontsize=8, ha=ha)
z = np.polyfit(ae_mean.values, target.values, 1)
xs = np.linspace(ae_mean.min() - 0.05, ae_mean.max() + 0.05, 50)
ax.plot(xs, np.polyval(z, xs), "--", c="gray", alpha=0.7, zorder=1)
rho, prho = spearmanr(ae_mean, target)
r, pr = pearsonr(ae_mean, target)
ax.set_xlabel("AE-suite mean accuracy at 1m\n"
              "(noisy_copy + compress + reverse_copy)")
ax.set_ylabel("50M LM wikitext\\_103 CE (lower=better)")
ax.set_title("(a) Autoencoding subset alone predicts LM ranking\n"
             f"Spearman $\\rho$={rho:.2f} (p={prho:.3f}),  "
             f"Pearson $r$={r:.2f} (p={pr:.3f})", fontsize=10)
ax.grid(alpha=0.3)
xpad = (ae_mean.max() - ae_mean.min()) * 0.10
ypad = (target.max() - target.min()) * 0.08
ax.set_xlim(ae_mean.min() - xpad, ae_mean.max() + xpad * 2.2)
ax.set_ylim(target.min() - ypad, target.max() + ypad)

# -------- Panel (b): per-AE-task ρ --------
ax = fig.add_subplot(gs[1])
per_task = []
for t in AE_TASKS:
    rho_t, p_t = spearmanr(WIDE.loc[t], target)
    per_task.append((t, float(rho_t), float(p_t)))
y = np.arange(len(AE_TASKS))
rhos = [pt[1] for pt in per_task]
ax.barh(y, rhos, color="#2ca02c", edgecolor="k", lw=0.5, alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels([pt[0].replace("_", "\\_") for pt in per_task], fontsize=10)
# p-value text to the right of each bar (outside the fill, x=0 area).
for yi, (_, rv, pv) in zip(y, per_task):
    ax.text(0.02, yi, f"$\\rho$={rv:+.2f},  p={pv:.3f}",
            ha="left", va="center", fontsize=9, color="black")
ax.axvline(rho, ls="--", c="k", lw=1.2,
           label=f"AE-mean $\\rho$={rho:.2f}")
ax.set_xlabel("Spearman $\\rho$ vs wikitext\\_103 CE (50M, n=9)")
ax.set_title("(b) Per-AE-task screening correlation", fontsize=10)
ax.invert_yaxis()
# Tight x-limits — bars all negative, leave headroom for the per-task text
xmin = min(rhos) - 0.05
ax.set_xlim(xmin, 0.45)
ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
ax.grid(axis="x", alpha=0.3)

plt.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
print(f"wrote {FIG_OUT}")
print(f"AE-mean ρ={rho:.3f} p={prho:.3f}")
