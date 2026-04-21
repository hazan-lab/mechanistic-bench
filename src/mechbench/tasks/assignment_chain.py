"""assignment_chain — Variables assigned from constants OR from earlier variables; emit value of a queried name.

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


def assignment_chain(rng, batch, spec: TaskSpec, n_vars: int = 6):
    """Variables are assigned either from a constant or from another
    variable; at the end emit the value of a query variable.

    Layout: [BOS, x1 MARKER v1, x2 MARKER <x1 or v2>, ..., SEP, q, ans]
    """
    T = spec.seq_len
    key_alpha = max(n_vars, spec.n_content // 3)
    val_alpha = spec.n_content - key_alpha
    if val_alpha < 2:
        raise ValueError("vocab too small for assignment_chain")
    used = 1 + 3 * n_vars + 1 + 2
    if used > T:
        raise ValueError("seq_len too short for assignment_chain")
    key_lo = CONTENT_LO
    val_lo = CONTENT_LO + key_alpha
    recs, mask = _empty(batch, T)
    for b in range(batch):
        keys = rng.permutation(key_alpha)[:n_vars] + key_lo
        constants = rng.integers(val_lo, val_lo + val_alpha, size=n_vars, dtype=np.int32)
        # each x_i either binds to its constant or to x_{i-1}'s value (chain)
        # resolved values per variable
        resolved = np.zeros(n_vars, dtype=np.int32)
        resolved[0] = constants[0]
        rhs = np.zeros(n_vars, dtype=np.int32)
        rhs[0] = constants[0]
        for i in range(1, n_vars):
            if rng.random() < 0.5:
                rhs[i] = constants[i]
                resolved[i] = constants[i]
            else:
                # bind to one of the earlier variable names
                prev = int(rng.integers(0, i))
                rhs[i] = keys[prev]
                resolved[i] = resolved[prev]
        pos = 1
        for i in range(n_vars):
            recs[b, pos] = keys[i]
            recs[b, pos + 1] = MARKER
            recs[b, pos + 2] = rhs[i]
            pos += 3
        recs[b, pos] = SEP
        pos += 1
        q_idx = int(rng.integers(0, n_vars))
        recs[b, pos] = keys[q_idx]
        recs[b, pos + 1] = resolved[q_idx]
        mask[b, pos] = True
    return recs, mask
