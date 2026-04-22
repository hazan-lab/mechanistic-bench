"""video_cell_mode — Modal cell value at (r, c) aggregated across n_frames frames.

Extracted from the wave-6 video sweep: this task trains above its modal
baseline while remaining below saturation on a 1M-param transformer on
short stacked-grid (video) inputs.
"""

from __future__ import annotations

import numpy as np

from .base import BOS, CONTENT_LO, MARKER, PAD, QUERY, SEP, TaskSpec


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    recs = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    recs[:, 0] = BOS
    return recs, mask


def video_cell_mode(rng, batch, spec: TaskSpec, side: int = 4, n_frames: int = 5):
    """Mode of the value at (r, c) across ``n_frames`` frames.

    Unique mode is enforced by planting ceil(n_frames/2) copies.
    Layout: [BOS, f1, MARKER, f2, MARKER, ..., SEP, r, c, mode]
    """
    T = spec.seq_len
    grid_size = side * side
    if spec.n_content < side + 4:
        raise ValueError("vocab too small for video_cell_mode")
    used = 1 + n_frames * grid_size + (n_frames - 1) + 1 + 3
    if used > T:
        raise ValueError("seq_len too short for video_cell_mode")
    coord_lo = CONTENT_LO
    val_lo = CONTENT_LO + side
    val_alpha = spec.n_content - side
    recs, mask = _empty(batch, T)
    for b in range(batch):
        frames = rng.integers(val_lo, val_lo + val_alpha, size=(n_frames, side, side), dtype=np.int32)
        r = int(rng.integers(0, side))
        c = int(rng.integers(0, side))
        mode_v = int(rng.integers(val_lo, val_lo + val_alpha))
        n_mode = n_frames // 2 + 1
        plant_idx = rng.choice(n_frames, size=n_mode, replace=False)
        # ensure non-planted frames at (r, c) != mode_v
        for f in range(n_frames):
            if f in plant_idx:
                frames[f, r, c] = mode_v
            else:
                while int(frames[f, r, c]) == mode_v:
                    frames[f, r, c] = int(rng.integers(val_lo, val_lo + val_alpha))
        pos = 1
        for i in range(n_frames):
            recs[b, pos : pos + grid_size] = frames[i].reshape(-1)
            pos += grid_size
            if i < n_frames - 1:
                recs[b, pos] = MARKER
                pos += 1
        recs[b, pos] = SEP
        pos += 1
        recs[b, pos] = coord_lo + r
        recs[b, pos + 1] = coord_lo + c
        recs[b, pos + 2] = mode_v
        mask[b, pos + 1] = True
    return recs, mask
