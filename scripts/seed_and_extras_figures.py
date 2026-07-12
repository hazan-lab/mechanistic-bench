#!/usr/bin/env python3
"""Generate figures for (a) the 5-seed 1M synthetic-task sweep and (b) the
single-seed 150M/50M LM experiments.

Seed data:  /scratch/gpfs/EHAZAN/tharuntk/seed_sweep/seed{1..5}/<task>-<arch>-1m/history.json
LM data:    /scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-*/history.json

Outputs PNGs + a tidy CSV under figures/seed_sweep/.
"""
import json, glob, os, collections, statistics, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED_ROOT = "/scratch/gpfs/EHAZAN/tharuntk/seed_sweep"
LM_ROOT   = "/scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs"
OUT       = "figures/seed_sweep"
os.makedirs(OUT, exist_ok=True)

# CVD-safe categorical palette (dataviz skill, fixed slot order)
PAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]
ARCHS = ["attn", "mamba", "mamba2", "alt_attn_mamba", "alt_attn_stu", "headwise", "stu"]
COLOR = {a: PAL[i] for i, a in enumerate(ARCHS)}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "figure.facecolor": "white", "axes.facecolor": "white",
})

# ---------- load seed data ----------
KNOWN = ["alt_attn_mamba", "alt_attn_stu", "attn", "headwise", "mamba2", "mamba", "stu"]
def split_arch(base):
    for a in KNOWN:
        if base.endswith("-" + a):
            return base[:-(len(a) + 1)], a
    raise ValueError(base)

# data[(task,arch)][seed] = final tok_acc
data = collections.defaultdict(dict)
tasks = set()
for s in range(1, 6):
    for hp in glob.glob(f"{SEED_ROOT}/seed{s}/*/history.json"):
        name = os.path.basename(os.path.dirname(hp))[:-3]  # strip -1m
        task, arch = split_arch(name)
        try:
            acc = json.load(open(hp))["history"][-1]["tok_acc"]
        except Exception:
            continue
        data[(task, arch)][s] = acc
        tasks.add(task)
tasks = sorted(tasks)
print(f"seed data: {len(tasks)} tasks x {len(ARCHS)} archs x 5 seeds")

# per-seed suite means -> seed mean/std per arch
seed_suite = collections.defaultdict(lambda: collections.defaultdict(list))
for (task, arch), sd in data.items():
    for s, acc in sd.items():
        seed_suite[arch][s].append(acc)
suite_mean, suite_std = {}, {}
for a in ARCHS:
    per_seed = [statistics.mean(v) for v in seed_suite[a].values()]
    suite_mean[a] = statistics.mean(per_seed)
    suite_std[a] = statistics.pstdev(per_seed)

order = sorted(ARCHS, key=lambda a: -suite_mean[a])

# ---------- Fig 1: per-arch suite mean tok_acc, seed error bars ----------
fig, ax = plt.subplots(figsize=(8, 4.6))
xs = np.arange(len(order))
bars = ax.bar(xs, [suite_mean[a] for a in order],
              yerr=[suite_std[a] for a in order], capsize=4,
              color=[COLOR[a] for a in order], width=0.68,
              error_kw=dict(ecolor=MUTED, lw=1.2))
for x, a in zip(xs, order):
    ax.text(x, suite_mean[a] + suite_std[a] + 0.012, f"{suite_mean[a]:.3f}",
            ha="center", va="bottom", fontsize=9.5, color=INK)
ax.set_xticks(xs)
ax.set_xticklabels([a.replace("_", "\n") for a in order], fontsize=9.5)
ax.set_ylabel("mean token accuracy  (63 tasks)")
ax.set_ylim(0, max(suite_mean.values()) + 0.09)
ax.set_title("1M synthetic mech-bench — suite mean over 5 seeds (±1 sd across seeds)",
             fontsize=11.5, color=INK, pad=10)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/seed_suite_mean.png", dpi=170); plt.close(fig)

# ---------- Fig 2: per-task heatmap of seed-mean tok_acc ----------
mat = np.full((len(tasks), len(ARCHS)), np.nan)
for i, t in enumerate(tasks):
    for j, a in enumerate(ARCHS):
        sd = data.get((t, a))
        if sd:
            mat[i, j] = statistics.mean(sd.values())
fig, ax = plt.subplots(figsize=(7.5, 15))
im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1)
ax.set_xticks(range(len(ARCHS)))
ax.set_xticklabels(ARCHS, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels(tasks, fontsize=6.5)
ax.set_title("Seed-mean token accuracy per task (1M)", fontsize=11, pad=8)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("tok_acc")
fig.tight_layout(); fig.savefig(f"{OUT}/seed_task_heatmap.png", dpi=150); plt.close(fig)

# ---------- write tidy CSV ----------
with open(f"{OUT}/seed_sweep_tokacc.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "arch", "seed_mean_tok_acc", "seed_std", "n_seeds"] +
               [f"seed{i}" for i in range(1, 6)])
    for t in tasks:
        for a in ARCHS:
            sd = data.get((t, a))
            if not sd: continue
            vals = [sd.get(i) for i in range(1, 6)]
            present = [v for v in vals if v is not None]
            w.writerow([t, a, round(statistics.mean(present), 5),
                        round(statistics.pstdev(present), 5), len(present)] +
                       [("" if v is None else round(v, 5)) for v in vals])

# ---------- LM (single-seed) figures ----------
def lm_last(run):
    hp = f"{LM_ROOT}/{run}/history.json"
    if not os.path.exists(hp): return None
    h = json.load(open(hp))
    last = h[-1] if isinstance(h, list) else h["history"][-1]
    return last

def get(last, key):
    return last.get(key) if last else None

C4 = "eval/eval/c4_en/Perplexity"
WT = "eval/eval/wikitext_103/Perplexity"

lm_archs = [("attn", "attn"), ("mamba2", "mamba2"),
            ("alt-attn-mamba2", "alt_attn_mamba")]
scales = [("150m", "150M / 3B tok"), ("50m", "50M / 1B tok")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, metric, mname in zip(axes, [C4, WT], ["C4", "WikiText-103"]):
    width = 0.35
    xs = np.arange(len(lm_archs))
    for k, (scode, slabel) in enumerate(scales):
        vals = []
        for run_a, col_a in lm_archs:
            last = lm_last(f"mechbench-{scode}-{run_a}")
            vals.append(get(last, metric) or np.nan)
        off = (k - 0.5) * width
        bars = ax.bar(xs + off, vals, width,
                      color=[COLOR[col_a] for _, col_a in lm_archs],
                      alpha=1.0 if k == 0 else 0.55,
                      edgecolor="white", linewidth=1.2, label=slabel)
        for x, v in zip(xs + off, vals):
            if not np.isnan(v):
                ax.text(x, v + max(vals) * 0.01, f"{v:.0f}", ha="center",
                        va="bottom", fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([a for a, _ in lm_archs], fontsize=9)
    ax.set_ylabel(f"{mname} perplexity  (↓)")
    ax.set_title(f"{mname} perplexity — single seed", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
# legend proxy: solid=150M, faded=50M
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(fc="#888", label="150M / 3B tok"),
                        Patch(fc="#888", alpha=0.55, label="50M / 1B tok")],
               frameon=False, fontsize=9)
fig.suptitle("LM perplexity by scale (no seed replicates — single run each)",
             fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/lm_perplexity_by_scale.png",
                                dpi=170, bbox_inches="tight"); plt.close(fig)

print("wrote:")
for f in sorted(os.listdir(OUT)):
    print("  ", os.path.join(OUT, f))
