"""Validate that the 10m hard-mode task configs generate valid batches."""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

from mechbench.tasks.base import TaskSpec
from mechbench.tasks.registry import TASK_REGISTRY, get_task


def modal_baseline(records, mask):
    labels = records[:, 1:][mask]
    if len(labels) == 0:
        return 0.0, 0
    counts = np.bincount(labels)
    return float(counts.max()) / len(labels), len(labels)


def main(cfg_path: str):
    cfg = json.loads(Path(cfg_path).read_text())
    rng = np.random.default_rng(0)
    bad = []
    for task, spec_cfg in cfg.items():
        if task.startswith("_"):
            continue
        base = TASK_REGISTRY[task]
        params = spec_cfg.get("task_params", {})
        fn = partial(base, **params) if params else base
        T = spec_cfg["seq_len"]
        V = spec_cfg["vocab_size"]
        spec = TaskSpec(name=task, seq_len=T, vocab_size=V)
        try:
            recs, mask = fn(rng, 256, spec)
            modal, n = modal_baseline(recs, mask)
            print(f"{task:26s}  T={T}  V={V}  n_labels={n:5d}  modal={modal:.3f}  params={params}")
            if modal > 0.6:
                bad.append((task, f"modal too high: {modal:.3f}"))
        except Exception as e:
            print(f"{task:26s}  ERROR: {type(e).__name__}: {e}")
            bad.append((task, str(e)))
    if bad:
        print("\nBAD:", bad)
        sys.exit(1)
    print("\nALL OK")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/tasks_10m_hard.json"
    main(cfg_path)
