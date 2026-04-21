"""union_lookup — Two disjoint key-value tables; retrieve value from whichever has the key.

Extracted from the wave-3 candidate sweep where this task trained above
its modal baseline while staying below saturation on a 1M-param
transformer, qualifying it as a non-saturating mechanistic probe.
"""

from __future__ import annotations

import numpy as np

from .base import BOS, CONTENT_LO, MARKER, PAD, QUERY, SEP, TaskSpec


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    records = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    records[:, 0] = BOS
    return records, mask


def union_lookup(rng, batch, spec: TaskSpec, n_pairs: int = 8):
    """Two key-value tables (disjoint key sets). Given a query key, emit its
    value (found in exactly one of the two tables).

    Layout: [BOS, A: k_1, v_1, ..., MARKER, B: k_1, v_1, ..., SEP, q, ans]
    """
    T = spec.seq_len
    # need 2*2*n_pairs keys disjoint + values from disjoint alphabet
    if spec.n_content < 2 * n_pairs + 2:
        raise ValueError("vocab too small for union_lookup")
    if 1 + 4 * n_pairs + 3 + 2 > T:
        raise ValueError("seq_len too short for union_lookup")
    key_lo = CONTENT_LO
    val_lo = CONTENT_LO + 2 * n_pairs  # shared value alphabet
    val_alpha = spec.n_content - 2 * n_pairs
    if val_alpha < 2:
        raise ValueError("value alphabet too small")
    recs, mask = _empty(batch, T)
    for b in range(batch):
        all_keys = np.arange(2 * n_pairs) + key_lo
        rng.shuffle(all_keys)
        keys_a = all_keys[:n_pairs]
        keys_b = all_keys[n_pairs:]
        vals_a = rng.integers(0, val_alpha, size=n_pairs, dtype=np.int32) + val_lo
        vals_b = rng.integers(0, val_alpha, size=n_pairs, dtype=np.int32) + val_lo
        pos = 1
        for i in range(n_pairs):
            recs[b, pos] = keys_a[i]
            recs[b, pos + 1] = vals_a[i]
            pos += 2
        recs[b, pos] = MARKER
        pos += 1
        for i in range(n_pairs):
            recs[b, pos] = keys_b[i]
            recs[b, pos + 1] = vals_b[i]
            pos += 2
        recs[b, pos] = SEP
        pos += 1
        # query: 50/50 from A or B
        from_a = int(rng.integers(0, 2))
        if from_a:
            idx = int(rng.integers(0, n_pairs))
            recs[b, pos] = keys_a[idx]
            recs[b, pos + 1] = vals_a[idx]
        else:
            idx = int(rng.integers(0, n_pairs))
            recs[b, pos] = keys_b[idx]
            recs[b, pos + 1] = vals_b[idx]
        mask[b, pos] = True
    return recs, mask
