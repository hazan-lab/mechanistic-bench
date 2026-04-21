"""variable_lookup — Key/value store with MARKER as '='; multiple queries after SEP.

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


def variable_lookup(rng, batch, spec: TaskSpec, n_vars: int = 8, n_queries: int = 3):
    """K/V store with MARKER as the "=" token; multiple queries at end.

    Layout: [BOS, v1 MARKER a1, v2 MARKER a2, ..., SEP, q1, a1', ..., qK, aK']
    """
    T = spec.seq_len
    key_alpha = spec.n_content // 2
    val_alpha = spec.n_content - key_alpha
    if n_vars > key_alpha or val_alpha < 2:
        raise ValueError("vocab too small for variable_lookup")
    used = 1 + 3 * n_vars + 1 + 2 * n_queries
    if used > T:
        raise ValueError("seq_len too short for variable_lookup")
    key_lo = CONTENT_LO
    val_lo = CONTENT_LO + key_alpha
    recs, mask = _empty(batch, T)
    for b in range(batch):
        keys = rng.permutation(key_alpha)[:n_vars] + key_lo
        vals = rng.integers(val_lo, val_lo + val_alpha, size=n_vars, dtype=np.int32)
        pos = 1
        for i in range(n_vars):
            recs[b, pos] = keys[i]
            recs[b, pos + 1] = MARKER
            recs[b, pos + 2] = vals[i]
            pos += 3
        recs[b, pos] = SEP
        pos += 1
        q_idx = rng.integers(0, n_vars, size=n_queries)
        for j in range(n_queries):
            recs[b, pos] = keys[q_idx[j]]
            recs[b, pos + 1] = vals[q_idx[j]]
            mask[b, pos] = True
            pos += 2
    return recs, mask
