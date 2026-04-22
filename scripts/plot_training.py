"""Plot train loss and train perplexity per task, one line per architecture.

Reads the ``train_log.json`` files produced by ``scripts/train.py`` under the
sweep's output directory (one run dir per ``<task>-<arch>-<scale>``) and writes
two PNGs per task into ``figures/train_loss/`` and ``figures/train_ppl/``.

Example
-------
    PYTHONPATH=/scratch/gpfs/EHAZAN/tharuntk/pysite \\
    uv run python scripts/plot_training.py \\
        --runs_dir /scratch/gpfs/EHAZAN/tharuntk/mechbench/runs_20260420_212136 \\
        --scale 1m
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Consistent arch ordering + colors across every plot so the visual key is stable.
ARCHS = ["attn", "mamba", "lstm", "alt_attn_mamba", "headwise", "stu"]
ARCH_COLORS = {
    "attn":           "#1f77b4",
    "mamba":          "#d62728",
    "lstm":           "#2ca02c",
    "alt_attn_mamba": "#9467bd",
    "headwise":       "#ff7f0e",
    "stu":            "#17becf",
}


def _load_curves(runs_dir: Path, scale: str) -> dict[str, dict[str, list[dict]]]:
    """Return {task: {arch: [{step, train_loss, train_ppl, ...}, ...]}}."""
    out: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    suffix = f"-{scale}"
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.endswith(suffix):
            continue
        stem = run_dir.name[: -len(suffix)]
        # stem is task-arch; arch is the last token matching our known set
        arch = None
        for a in ARCHS:
            if stem.endswith(f"-{a}"):
                arch = a
                task = stem[: -len(f"-{a}")]
                break
        if arch is None:
            continue
        train_log = run_dir / "train_log.json"
        if not train_log.exists():
            continue
        data = json.loads(train_log.read_text())
        out[task][arch] = data["train_log"]
    return out


def _plot_metric(
    task: str,
    arch_curves: dict[str, list[dict]],
    metric: str,
    ylabel: str,
    out_path: Path,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for arch in ARCHS:
        curve = arch_curves.get(arch)
        if not curve:
            continue
        steps = [p["step"] for p in curve]
        ys = [p[metric] for p in curve]
        ax.plot(steps, ys, label=arch, color=ARCH_COLORS[arch], linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{task}  —  {ylabel} vs step  (1M params)")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", required=True, type=Path)
    p.add_argument("--scale", default="1m")
    p.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = p.parse_args()

    curves = _load_curves(args.runs_dir, args.scale)
    if not curves:
        raise SystemExit(f"no train_log.json found under {args.runs_dir}")

    tasks = sorted(curves.keys())
    print(f"tasks with data: {len(tasks)}")
    loss_dir = args.out_dir / "train_loss"
    ppl_dir = args.out_dir / "train_ppl"
    for task in tasks:
        arch_curves = curves[task]
        _plot_metric(task, arch_curves, "train_loss", "train loss", loss_dir / f"{task}.png")
        _plot_metric(task, arch_curves, "train_ppl", "train perplexity", ppl_dir / f"{task}.png", log_y=True)
    print(f"wrote {len(tasks)} loss plots -> {loss_dir}")
    print(f"wrote {len(tasks)} ppl plots  -> {ppl_dir}")


if __name__ == "__main__":
    main()
