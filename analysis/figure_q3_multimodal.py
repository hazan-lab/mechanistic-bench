"""Figure: multimodal-subset screening (Q3).

Splits the suite into:
  - text-only tasks (no spatial / continuous primitives)
  - multimodal tasks (grid + video, plus discriminating continuous probes)

For each subset, correlates the per-arch mech-suite mean accuracy with
50M LM cross-entropy on wikitext_103 (held-out). Both subsets correlate at
p<0.05 with n=9 architectures.

Layout fix: text panel had alt-Attn/{Mamba2,STU,Mamba} labels overlapping
on the right; multimodal panel had a 5-arch cluster on the lower-left
(STU, Mamba, headwise variants) where labels collided. Hand-placed
per-panel offsets eliminate the overlaps.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent
FIG_OUT = Path("/home/tt6444/mechanistic-bench/icml_workshop/figures/q3_multimodal.png")
FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

WIDE = pd.read_csv(ROOT / "unified_1m.csv").set_index("task")
LM = pd.read_csv(ROOT / "lm_ce.csv")
LM50 = LM[LM.scale == "50m"].set_index("arch")

keep_archs = ['attn', 'alt_attn_mamba', 'alt_attn_mamba2', 'alt_attn_stu', 'stu',
              'headwise', 'headwise_mamba2', 'headwise_stu', 'mamba']
WIDE = WIDE[keep_archs].dropna()

DISC_MULTI = ['grid_multihop', 'grid_retrieval', 'grid_three_coord', 'grid_two_coord',
              'video_cell_mode', 'video_frame_retrieval']
text_tasks = [t for t in WIDE.index if t not in DISC_MULTI
              and t not in ['col_parity', 'delayed_echo', 'patch_match',
                            'nearest_key', 'piecewise_denoise']]
text_mean = WIDE.loc[text_tasks].mean(axis=0)
multi_mean = WIDE.loc[DISC_MULTI].mean(axis=0)

ARCH_LABELS = {
    "attn": "Attn", "stu": "STU", "mamba": "Mamba",
    "alt_attn_mamba": "alt-Attn/Mamba", "alt_attn_mamba2": "alt-Attn/Mamba2",
    "alt_attn_stu": "alt-Attn/STU",
    "headwise": "head-Attn/Mamba", "headwise_mamba2": "head-Attn/Mamba2",
    "headwise_stu": "head-Attn/STU",
}
target = LM50.loc[keep_archs, "wikitext_103_CE"]

# Hand-placed offsets per panel.
OFF_TEXT = {
    "attn":             (8, 4),
    "stu":              (8, 4),
    "mamba":            (8, 4),
    "alt_attn_mamba":   (8, -10),
    "alt_attn_mamba2":  (8, 8),
    "alt_attn_stu":     (-10, -12),
    "headwise":         (8, 8),
    "headwise_mamba2":  (8, -10),
    "headwise_stu":     (-8, 8),
}
# Multimodal panel: 5 archs cluster at x≈0.08-0.10 — use leader arrows so
# their labels don't overlap. Format: (dx_pts, dy_pts, use_arrow).
OFF_MULTI = {
    "stu":              (40, 12, True),
    "headwise_stu":     (50, 4, True),
    "headwise":         (60, -4, True),
    "headwise_mamba2":  (70, -14, True),
    "mamba":            (50, -22, True),
    "alt_attn_mamba2":  (8, 6, False),
    "attn":             (8, 4, False),
    "alt_attn_stu":     (8, 4, False),
    "alt_attn_mamba":   (-8, 8, False),
}


def panel(ax, name, x_series, color, off):
    ax.scatter(x_series, target, s=85, c=color, edgecolors="k",
               linewidth=1, zorder=3)
    for a in keep_archs:
        spec = off.get(a, (5, 4))
        if len(spec) == 3:
            dx, dy, use_arrow = spec
        else:
            dx, dy = spec
            use_arrow = False
        ha = "left" if dx >= 0 else "right"
        kw = dict(textcoords="offset points", xytext=(dx, dy),
                  fontsize=8, ha=ha, va="center")
        if use_arrow:
            kw["arrowprops"] = dict(arrowstyle="-", color="0.5", lw=0.6,
                                    shrinkA=0, shrinkB=2)
        ax.annotate(ARCH_LABELS.get(a, a), (x_series[a], target[a]), **kw)
    z = np.polyfit(x_series.values, target.values, 1)
    xs = np.linspace(x_series.min() - 0.04, x_series.max() + 0.04, 50)
    ax.plot(xs, np.polyval(z, xs), "--", c="gray", alpha=0.7, zorder=1)
    rho, prho = spearmanr(x_series, target)
    r, pr = pearsonr(x_series, target)
    ax.set_title(f"{name}\n"
                 f"Spearman $\\rho$={rho:.2f} (p={prho:.3f}),  "
                 f"Pearson $r$={r:.2f} (p={pr:.3f})", fontsize=10)
    ax.set_xlabel("mech-suite mean accuracy at 1m")
    ax.grid(alpha=0.3)
    xpad = (x_series.max() - x_series.min()) * 0.10
    ax.set_xlim(x_series.min() - xpad, x_series.max() + xpad * 1.6)


fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True,
                          gridspec_kw={"wspace": 0.10})
panel(axes[0], f"Text-only suite ({len(text_tasks)} tasks)",
      text_mean, "#1f77b4", OFF_TEXT)
panel(axes[1], f"Multimodal-only (grid + video, {len(DISC_MULTI)} tasks)",
      multi_mean, "#ff7f0e", OFF_MULTI)
axes[0].set_ylabel("50M LM wikitext\\_103 CE (lower=better)")

# Give the y-axis a touch of padding so labels above/below points don't get clipped.
ypad = (target.max() - target.min()) * 0.10
axes[0].set_ylim(target.min() - ypad, target.max() + ypad)

plt.suptitle("Multimodal probes alone predict held-out 50M-LM ranking "
             "(n=9 architectures)\n"
             "Both text-only and multimodal-only subsets correlate at p<0.05",
             y=1.04, fontsize=11)
plt.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
print(f"wrote {FIG_OUT}")
