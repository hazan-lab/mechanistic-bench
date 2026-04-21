"""grid_multihop — One 2D hop: read cell at (r, c), treat its value as a row id, emit the cell in that row at column c.

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


def grid_multihop(rng, batch, spec: TaskSpec):
    """The grid cells are themselves row-coord ids. Starting from (r0, c0):
    read value v0 = grid[r0, c0] (a row id); then emit grid[v0, c0] (i.e.
    jump to the row that v0 points to, same column). One 2D hop.

    Layout: [BOS, grid, SEP, r0, c0, answer]
    """
    T = spec.seq_len
    side = _grid_side(T, overhead=4, cap=8)
    H = W = side
    grid_size = H * W
    if spec.n_content < side:
        raise ValueError("vocab too small for grid_multihop")
    # grid cells are row-ids in [0, side), encoded as CONTENT_LO + id
    val_lo = CONTENT_LO
    coord_lo = CONTENT_LO  # same as val_lo — ok since both live in [0, side)
    recs, mask = _empty(batch, T)
    for b in range(batch):
        grid = rng.integers(0, side, size=(H, W), dtype=np.int32)
        recs[b, 1 : 1 + grid_size] = val_lo + grid.reshape(-1)
        sep_pos = 1 + grid_size
        recs[b, sep_pos] = SEP
        r0 = int(rng.integers(0, H))
        c0 = int(rng.integers(0, W))
        v0 = int(grid[r0, c0])
        recs[b, sep_pos + 1] = coord_lo + r0
        recs[b, sep_pos + 2] = coord_lo + c0
        recs[b, sep_pos + 3] = val_lo + int(grid[v0, c0])
        mask[b, sep_pos + 2] = True
    return recs, mask
