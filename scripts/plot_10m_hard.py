"""Plots for the 10m hard-mode multi-arch sweep.

Walks ``<sweep_root>`` expecting two possible layouts for a given (task, arch):

    <root>/<task>-<arch>-10m-hard/{history.json, train_log.json}      # legacy flat (attn from PR #31)
    <root>/<arch>/<task>-<arch>-10m-hard/{history.json, train_log.json}  # per-arch subdirs

Produces, under ``--out``:

  - ``per_task_tok_acc.png``       — 29-panel grid of eval tok_acc vs step, one line per arch.
  - ``per_task_train_loss.png``    — 29-panel grid of training loss vs step.
  - ``per_task_eval_ppl.png``      — 29-panel grid of eval perplexity vs step.
  - ``avg_tok_acc_vs_step.png``    — mean tok_acc over all tasks at each eval step, per arch.
  - ``final_tok_acc_heatmap.png``  — final-step tok_acc, arch × task.
  - ``final_tok_acc_bar.png``      — mean final-step tok_acc per arch (bars + per-task strip).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARCH_ORDER = ["attn", "lstm", "stu", "mamba", "alt_attn_mamba", "headwise"]
ARCH_COLORS = {
    "attn": "#1f77b4",
    "lstm": "#d62728",
    "stu": "#2ca02c",
    "mamba": "#9467bd",
    "alt_attn_mamba": "#ff7f0e",
    "headwise": "#8c564b",
}


def discover_runs(sweep_root: Path) -> dict[tuple[str, str], Path]:
    """Return {(task, arch): run_dir} by walking both layouts."""
    runs: dict[tuple[str, str], Path] = {}
    for child in sorted(sweep_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.endswith("-10m-hard"):
            # legacy flat: <task>-<arch>-10m-hard
            task, arch, _, _ = child.name.rsplit("-", 3)
            runs[(task, arch)] = child
        elif child.name in ARCH_ORDER:
            # new: <arch>/<task>-<arch>-10m-hard
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and sub.name.endswith("-10m-hard"):
                    task, arch, _, _ = sub.name.rsplit("-", 3)
                    runs[(task, arch)] = sub
    return runs


def load_history(run_dir: Path) -> dict[str, np.ndarray] | None:
    hp = run_dir / "history.json"
    if not hp.exists():
        return None
    data = json.loads(hp.read_text())
    hist = data["history"] if isinstance(data, dict) and "history" in data else data
    if not hist:
        return None
    out = {}
    for k in ("step", "tok_acc", "eval_loss", "eval_ppl"):
        out[k] = np.array([h.get(k, np.nan) for h in hist], dtype=float)
    return out


def load_train_log(run_dir: Path) -> dict[str, np.ndarray] | None:
    lp = run_dir / "train_log.json"
    if not lp.exists():
        return None
    data = json.loads(lp.read_text())
    log = data["train_log"] if isinstance(data, dict) and "train_log" in data else data
    if not log:
        return None
    out = {}
    for k in ("step", "train_loss", "train_ppl"):
        out[k] = np.array([h.get(k, np.nan) for h in log], dtype=float)
    return out


def _grid_shape(n: int) -> tuple[int, int]:
    ncols = 6 if n > 20 else 5
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def plot_per_task(
    runs: dict[tuple[str, str], Path],
    tasks: list[str],
    archs: list[str],
    metric_source: str,   # "history" or "train_log"
    metric: str,          # tok_acc | eval_loss | eval_ppl | train_loss | train_ppl
    ylabel: str,
    out_path: Path,
    logy: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    nrows, ncols = _grid_shape(len(tasks))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)
    loader = load_history if metric_source == "history" else load_train_log
    for i, task in enumerate(tasks):
        ax = axes[i]
        for arch in archs:
            run_dir = runs.get((task, arch))
            if run_dir is None:
                continue
            d = loader(run_dir)
            if d is None or metric not in d:
                continue
            ax.plot(d["step"], d[metric], color=ARCH_COLORS.get(arch, "k"),
                    lw=1.1, label=arch)
        ax.set_title(task, fontsize=8)
        ax.tick_params(labelsize=7)
        if logy:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
    for j in range(len(tasks), len(axes)):
        axes[j].axis("off")
    # shared legend in unused slot or top right
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{ylabel} vs step — 10m hard-mode, per task", y=1.04, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_avg_tok_acc(runs, tasks, archs, out_path: Path) -> None:
    """Average tok_acc across tasks at each eval checkpoint, one line per arch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for arch in archs:
        per_task_curves = []
        common_steps = None
        for task in tasks:
            run_dir = runs.get((task, arch))
            if run_dir is None:
                continue
            d = load_history(run_dir)
            if d is None:
                continue
            if common_steps is None:
                common_steps = d["step"]
                per_task_curves.append(d["tok_acc"])
            else:
                # pad/trim to common_steps length
                n = min(len(common_steps), len(d["step"]))
                if np.allclose(common_steps[:n], d["step"][:n]):
                    per_task_curves.append(d["tok_acc"][:n])
                    common_steps = common_steps[:n]
        if not per_task_curves:
            continue
        # make them the same length
        L = min(len(c) for c in per_task_curves)
        arr = np.stack([c[:L] for c in per_task_curves])
        mean = arr.mean(axis=0)
        ax.plot(common_steps[:L], mean, color=ARCH_COLORS.get(arch, "k"),
                lw=2.0, marker="o", ms=4, label=f"{arch} (n={arr.shape[0]})")
    ax.set_xlabel("step")
    ax.set_ylabel("mean tok_acc across tasks")
    ax.set_title("10m hard-mode — average eval tok_acc vs training step")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_heatmap(runs, tasks, archs, out_path: Path) -> None:
    """arch × task heatmap of final-step tok_acc."""
    mat = np.full((len(archs), len(tasks)), np.nan)
    for i, arch in enumerate(archs):
        for j, task in enumerate(tasks):
            run_dir = runs.get((task, arch))
            if run_dir is None:
                continue
            d = load_history(run_dir)
            if d is None:
                continue
            mat[i, j] = d["tok_acc"][-1]
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(tasks) + 4), 0.55 * len(archs) + 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=75, fontsize=8, ha="right")
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(archs, fontsize=10)
    for i in range(len(archs)):
        for j in range(len(tasks)):
            if np.isnan(mat[i, j]):
                continue
            txt = f"{mat[i, j]:.2f}"
            color = "white" if mat[i, j] < 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    ax.set_title("Final-step tok_acc (4000 steps) — 10m hard-mode")
    fig.colorbar(im, ax=ax, shrink=0.8, label="tok_acc")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_bar(runs, tasks, archs, out_path: Path) -> None:
    """Mean final tok_acc per arch (error bar = std over tasks), plus per-task strip."""
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(archs))
    means = []
    stds = []
    for i, arch in enumerate(archs):
        vals = []
        for task in tasks:
            run_dir = runs.get((task, arch))
            if run_dir is None:
                continue
            d = load_history(run_dir)
            if d is None:
                continue
            vals.append(d["tok_acc"][-1])
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals) if vals else 0)
        # per-task strip
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.3
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=ARCH_COLORS.get(arch, "k"), alpha=0.5, s=18, zorder=3)
    bars = ax.bar(xs, means, yerr=stds, capsize=5,
                  color=[ARCH_COLORS.get(a, "gray") for a in archs],
                  alpha=0.65, edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(archs, fontsize=10)
    ax.set_ylabel("final tok_acc (mean over tasks, ± std)")
    ax.set_title("Arch ranking — 10m hard-mode mean final tok_acc")
    for x, m in zip(xs, means):
        if not np.isnan(m):
            ax.text(x, m + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", default="/scratch/gpfs/EHAZAN/tharuntk/mechbench/sweep_10m_hard")
    p.add_argument("--out", default="figures/10m_multiarch")
    p.add_argument("--tasks_config", default="configs/tasks_10m_hard.json",
                   help="Used to determine task order. Non-underscore keys only.")
    args = p.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.tasks_config).read_text())
    tasks = [t for t in cfg.keys() if not t.startswith("_")]

    runs = discover_runs(sweep_root)
    present_archs = sorted({a for (_, a) in runs.keys()}, key=lambda x: ARCH_ORDER.index(x) if x in ARCH_ORDER else 99)
    print(f"Discovered {len(runs)} runs across archs: {present_archs}")

    plot_per_task(runs, tasks, present_archs, "history", "tok_acc",
                  "eval tok_acc", out_dir / "per_task_tok_acc.png", ylim=(0, 1))
    plot_per_task(runs, tasks, present_archs, "history", "eval_ppl",
                  "eval perplexity", out_dir / "per_task_eval_ppl.png", logy=True)
    plot_per_task(runs, tasks, present_archs, "train_log", "train_loss",
                  "train loss", out_dir / "per_task_train_loss.png")
    plot_avg_tok_acc(runs, tasks, present_archs, out_dir / "avg_tok_acc_vs_step.png")
    plot_final_heatmap(runs, tasks, present_archs, out_dir / "final_tok_acc_heatmap.png")
    plot_final_bar(runs, tasks, present_archs, out_dir / "final_tok_acc_bar.png")

    print(f"Wrote plots to {out_dir}/")


if __name__ == "__main__":
    main()
