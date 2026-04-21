"""sort_top2 — Emit the two largest body tokens in descending order.

Extracted from the wave-5 candidate sweep (algorithmic/sequence
primitives): this task trains above its modal baseline yet stays below
saturation on a 1M-param transformer.
"""

from __future__ import annotations

import numpy as np

from .base import BOS, CONTENT_LO, MARKER, PAD, QUERY, SEP, TaskSpec


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    recs = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    recs[:, 0] = BOS
    return recs, mask


def sort_top2(rng, batch, spec: TaskSpec, body_len: int = 48, alpha: int = 12):
    """Emit the top-2 tokens (descending, ties broken by smallest id).

    Layout: [BOS, body, SEP, top1, top2]
    Uniformly planting the top value so labels flat over [1, alpha).
    """
    T = spec.seq_len
    if body_len + 3 > T:
        raise ValueError("seq_len too short for sort_top2")
    if spec.n_content < alpha:
        raise ValueError("vocab too small for sort_top2")
    val_lo = CONTENT_LO
    recs, mask = _empty(batch, T)
    for b in range(batch):
        top1 = int(rng.integers(1, alpha))
        top2 = int(rng.integers(0, top1))
        body = rng.integers(0, top1, size=body_len, dtype=np.int32)
        # strip any >= top1
        # plant top1 once; plant top2 once
        p1 = int(rng.integers(0, body_len))
        body[p1] = top1
        p2 = int(rng.integers(0, body_len - 1))
        if p2 == p1:
            p2 = body_len - 1
        body[p2] = top2
        recs[b, 1 : 1 + body_len] = val_lo + body
        sep_pos = 1 + body_len
        recs[b, sep_pos] = SEP
        recs[b, sep_pos + 1] = val_lo + top1
        recs[b, sep_pos + 2] = val_lo + top2
        mask[b, sep_pos : sep_pos + 2] = True
    return recs, mask
