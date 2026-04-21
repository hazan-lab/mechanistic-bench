"""video_frame_retrieval — Retrieve cell value at (frame, row, col) from a stacked sequence of small grids.

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


def video_frame_retrieval(rng, batch, spec: TaskSpec, side: int = 4, n_frames: int = 3):
    """Coordinate-plus-frame retrieval across a short video.

    Layout: [BOS, f1, MARKER, f2, MARKER, f3, SEP, frame_idx, r, c, value]
    """
    T = spec.seq_len
    grid_size = side * side
    # frame alphabet: n_frames; coord alphabet: side
    if spec.n_content < max(side, n_frames) + 4:
        raise ValueError("vocab too small for video_frame_retrieval")
    used = 1 + n_frames * grid_size + (n_frames - 1) + 1 + 4
    if used > T:
        raise ValueError("seq_len too short for video_frame_retrieval")
    frame_lo = CONTENT_LO  # shares space with coord_lo since both are small ints
    coord_lo = CONTENT_LO
    val_lo = CONTENT_LO + max(side, n_frames)
    val_alpha = spec.n_content - max(side, n_frames)
    recs, mask = _empty(batch, T)
    for b in range(batch):
        frames = rng.integers(val_lo, val_lo + val_alpha, size=(n_frames, side, side), dtype=np.int32)
        pos = 1
        for i in range(n_frames):
            recs[b, pos : pos + grid_size] = frames[i].reshape(-1)
            pos += grid_size
            if i < n_frames - 1:
                recs[b, pos] = MARKER
                pos += 1
        recs[b, pos] = SEP
        pos += 1
        f = int(rng.integers(0, n_frames))
        r = int(rng.integers(0, side))
        c = int(rng.integers(0, side))
        recs[b, pos] = frame_lo + f
        recs[b, pos + 1] = coord_lo + r
        recs[b, pos + 2] = coord_lo + c
        recs[b, pos + 3] = int(frames[f, r, c])
        mask[b, pos + 2] = True
    return recs, mask
