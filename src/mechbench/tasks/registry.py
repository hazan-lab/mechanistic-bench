"""Task registry.

Each entry maps a task name → generator callable with signature
``(rng, batch_size, spec) -> (records, loss_mask)``.
"""

from __future__ import annotations

from functools import partial

from . import synthetic as S

TASK_REGISTRY: dict = {
    # retrieval
    "copy": S.copy,
    "copy_offset": S.copy_offset,
    "reverse_copy": S.reverse_copy,
    "induction": S.induction,
    "induction_gap": S.induction_gap,
    "multi_induction": S.multi_induction,
    "associative": S.associative,
    "short_associative": S.short_associative,
    "selective_copy": S.selective_copy,
    "needle": S.needle,
    # aggregation
    "counting": S.counting,
    "parity": S.parity,
    "cumulative_sum": S.cumulative_sum,
    "state_tracking": S.state_tracking,
    "mode": S.mode,
    "sort": S.sort_task,
}


def list_tasks() -> list[str]:
    return sorted(TASK_REGISTRY.keys())


def get_task(name: str):
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {list_tasks()}")
    return TASK_REGISTRY[name]
