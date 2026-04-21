"""Task registry.

Each entry maps a task name → generator callable with signature
``(rng, batch_size, spec) -> (records, loss_mask)``.
"""

from __future__ import annotations

from functools import partial

from . import batch_recall as BR
from . import continuous as C
from . import two_hop as T_two_
from . import mode_tagged as T_mode
from . import conditional_recall as T_cond
from . import first_vs_last as T_firs
from . import triple_recall as T_trip
from . import nested_lookup as T_nest
from . import three_hop as T_thre
from . import quad_recall as T_quad
from . import last_tagged as T_last
from . import hop_distance_bucket as T_hop_
from . import batch_two_hop as T_batc
from . import synthetic as S
from . import vision as V

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
    # extended reasoning
    "pattern_completion": S.pattern_completion,
    "compress": S.compress,
    "interleave": S.interleave,
    "longest_run": S.longest_run,
    "threshold": S.threshold,
    "token_transition": S.token_transition,
    "noisy_copy": S.noisy_copy,
    "running_max": S.running_max,
    "selective_parity": S.selective_parity,
    "multi_state_tracking": S.multi_state_tracking,
    "state_retrieve": S.state_retrieve,
    "copy_count": S.copy_count,
    # vision (raster-flattened 2D)
    "grid_retrieval": V.grid_retrieval,
    "col_parity": V.col_parity,
    "patch_match": V.patch_match,
    # continuous (tokenised real-valued signals)
    "delayed_echo": C.delayed_echo,
    "piecewise_denoise": C.piecewise_denoise,
    "nearest_key": C.nearest_key,
    # parallel retrieval
    "batch_recall": BR.batch_recall,
    "two_hop": T_two_.two_hop,
    "mode_tagged": T_mode.mode_tagged,
    "conditional_recall": T_cond.conditional_recall,
    "first_vs_last": T_firs.first_vs_last,
    "triple_recall": T_trip.triple_recall,
    "nested_lookup": T_nest.nested_lookup,
    "three_hop": T_thre.three_hop,
    "quad_recall": T_quad.quad_recall,
    "last_tagged": T_last.last_tagged,
    "hop_distance_bucket": T_hop_.hop_distance_bucket,
    "batch_two_hop": T_batc.batch_two_hop,
}


def list_tasks() -> list[str]:
    return sorted(TASK_REGISTRY.keys())


def get_task(name: str):
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {list_tasks()}")
    return TASK_REGISTRY[name]
