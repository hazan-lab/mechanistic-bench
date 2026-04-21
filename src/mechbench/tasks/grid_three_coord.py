"""grid_three_coord — Three parallel (row, col) queries against a flattened 2D grid; emit all three values.

Extracted from the wave-4 vision candidate sweep: this grid-based task
trains above its modal baseline yet stays below saturation on a 1M-param
transformer, qualifying it as a non-saturated vision mechanistic probe.
"""

from __future__ import annotations

import numpy as np

from .base import BOS, CONTENT_LO, MARKER, PAD, QUERY, SEP, TaskSpec


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    recs = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    recs[:, 0] = BOS
    return recs, mask


def _grid_side(T: int, overhead: int, cap: int = 12) -> int:
    side = int(np.floor(np.sqrt(max(1, T - overhead))))
    return max(4, min(side, cap))


def grid_three_coord(rng, batch, spec: TaskSpec):
    T = spec.seq_len
    side = _grid_side(T, overhead=12)
    H = W = side
    grid_size = H * W
    if spec.n_content < side + 4:
        raise ValueError("vocab too small for grid_three_coord")
    if 1 + grid_size + 1 + 3 * 3 > T:
        raise ValueError("seq_len too short for grid_three_coord")
    coord_lo = CONTENT_LO
    val_lo = CONTENT_LO + side
    val_hi = spec.vocab_size
    recs, mask = _empty(batch, T)
    grid = rng.integers(val_lo, val_hi, size=(batch, H, W), dtype=np.int32)
    recs[:, 1 : 1 + grid_size] = grid.reshape(batch, grid_size)
    sep_pos = 1 + grid_size
    recs[:, sep_pos] = SEP
    idx = np.arange(batch)
    for j in range(3):
        r = rng.integers(0, H, size=batch, dtype=np.int64)
        c = rng.integers(0, W, size=batch, dtype=np.int64)
        p = sep_pos + 1 + 3 * j
        recs[idx, p] = coord_lo + r.astype(np.int32)
        recs[idx, p + 1] = coord_lo + c.astype(np.int32)
        recs[idx, p + 2] = grid[idx, r, c]
        mask[:, p + 1] = True
    return recs, mask
