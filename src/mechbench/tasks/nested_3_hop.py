"""nested_3_hop — Three-level redirect chain k1 -> k2 -> k3 -> v.

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


def nested_3_hop(rng, batch, spec: TaskSpec, n_pairs: int = 6):
    """Three redirect tables: k1->k2, k2->k3, k3->v. Given query q=k1, emit v.

    Layout: [BOS, T1, SEP, T2, SEP, T3, SEP, q, ans]
    """
    T = spec.seq_len
    if spec.n_content < 4 * n_pairs:
        raise ValueError("vocab too small for nested_3_hop")
    if 1 + 6 * n_pairs + 3 + 2 > T:
        raise ValueError("seq_len too short for nested_3_hop")
    k1_lo = CONTENT_LO
    k2_lo = CONTENT_LO + n_pairs
    k3_lo = CONTENT_LO + 2 * n_pairs
    v_lo = CONTENT_LO + 3 * n_pairs
    recs, mask = _empty(batch, T)
    for b in range(batch):
        k1s = np.arange(n_pairs) + k1_lo
        k2_perm = (np.arange(n_pairs) + k2_lo).astype(np.int32)
        rng.shuffle(k2_perm)
        k3_perm = (np.arange(n_pairs) + k3_lo).astype(np.int32)
        rng.shuffle(k3_perm)
        vs = rng.integers(v_lo, v_lo + n_pairs, size=n_pairs, dtype=np.int32)
        # table 1: k1 -> k2
        order = rng.permutation(n_pairs)
        pos = 1
        for i in order:
            recs[b, pos] = k1s[i]
            recs[b, pos + 1] = k2_perm[i]
            pos += 2
        recs[b, pos] = SEP
        pos += 1
        # table 2: enumerate k2 values -> k3
        k2_all = (np.arange(n_pairs) + k2_lo).astype(np.int32)
        k2_to_k3 = {int(k2_perm[i]): int(k3_perm[i]) for i in range(n_pairs)}
        order2 = rng.permutation(n_pairs)
        for i in order2:
            k2 = int(k2_all[i])
            recs[b, pos] = k2
            recs[b, pos + 1] = k2_to_k3.get(k2, int(k3_perm[0]))
            pos += 2
        recs[b, pos] = SEP
        pos += 1
        # table 3: k3 -> v
        k3_all = (np.arange(n_pairs) + k3_lo).astype(np.int32)
        k3_to_v = {int(k3_perm[i]): int(vs[i]) for i in range(n_pairs)}
        order3 = rng.permutation(n_pairs)
        for i in order3:
            k3 = int(k3_all[i])
            recs[b, pos] = k3
            recs[b, pos + 1] = k3_to_v.get(k3, int(vs[0]))
            pos += 2
        recs[b, pos] = SEP
        pos += 1
        # query
        q_idx = int(rng.integers(0, n_pairs))
        recs[b, pos] = int(k1s[q_idx])
        recs[b, pos + 1] = int(vs[q_idx])
        mask[b, pos] = True
    return recs, mask
