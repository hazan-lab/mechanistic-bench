"""Validate that the 10m hard-mode task configs generate valid batches."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import yaml

from mechbench.tasks.base import TaskSpec
from mechbench.tasks.registry import TASK_REGISTRY


def modal_baseline(records, mask):
    labels = records[:, 1:][mask]
    if len(labels) == 0:
        return 0.0, 0
    counts = np.bincount(labels)
    return float(counts.max()) / len(labels), len(labels)


def main(cfg_path: str):
    doc = yaml.safe_load(Path(cfg_path).read_text()) or {}
    defaults = doc.get("defaults", {}) or {}
    tasks = doc.get("tasks", {}) or {}
    rng = np.random.default_rng(0)
    bad = []
    for task, entry in tasks.items():
        entry = entry or {}
        T = entry.get("seq_len", defaults.get("seq_len"))
        V = entry.get("vocab_size", defaults.get("vocab_size"))
        params = dict(defaults.get("task_params") or {})
        params.update(entry.get("task_params") or {})
        base = TASK_REGISTRY[task]
        fn = partial(base, **params) if params else base
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
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/scale_10m/tasks.yaml"
    main(cfg_path)
