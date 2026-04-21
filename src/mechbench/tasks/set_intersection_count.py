"""set_intersection_count — Two multisets separated by MARKER; emit bucketed |A intersect B|.

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


def set_intersection_count(rng, batch, spec: TaskSpec, set_size: int = 10, alpha: int = 16, max_count: int = 8):
    """Two multisets separated by MARKER; emit |A ∩ B| (count of shared
    values) bucketed into max_count bins.

    Layout: [BOS, A..., MARKER, B..., SEP, count]
    """
    T = spec.seq_len
    used = 1 + 2 * set_size + 1 + 1 + 1
    if used > T:
        raise ValueError("seq_len too short for set_intersection_count")
    if spec.n_content < alpha + max_count:
        raise ValueError("vocab too small for set_intersection_count")
    val_lo = CONTENT_LO
    count_lo = val_lo + alpha
    recs, mask = _empty(batch, T)
    for b in range(batch):
        # sample target count uniformly in [0, min(set_size, max_count-1)]
        target = int(rng.integers(0, min(set_size, max_count - 1) + 1))
        shared = rng.choice(alpha, size=target, replace=False) if target > 0 else np.array([], dtype=np.int64)
        # A = shared + (set_size - target) unique non-shared
        non_shared = np.setdiff1d(np.arange(alpha), shared)
        a_only = rng.choice(non_shared, size=set_size - target, replace=False)
        A = np.concatenate([shared, a_only])
        # B = shared + different non-shared elements
        b_only_pool = np.setdiff1d(non_shared, a_only)
        if len(b_only_pool) < set_size - target:
            # allow repeats in b_only from non_shared
            b_only = rng.choice(non_shared, size=set_size - target, replace=True)
            # ensure no b_only element equals any a_only element
            for i in range(len(b_only)):
                while b_only[i] in a_only:
                    b_only[i] = int(rng.choice(non_shared))
        else:
            b_only = rng.choice(b_only_pool, size=set_size - target, replace=False)
        B = np.concatenate([shared, b_only])
        rng.shuffle(A)
        rng.shuffle(B)
        pos = 1
        recs[b, pos : pos + set_size] = val_lo + A.astype(np.int32)
        pos += set_size
        recs[b, pos] = MARKER
        pos += 1
        recs[b, pos : pos + set_size] = val_lo + B.astype(np.int32)
        pos += set_size
        recs[b, pos] = SEP
        pos += 1
        recs[b, pos] = count_lo + target
        mask[b, pos - 1] = True
    return recs, mask
