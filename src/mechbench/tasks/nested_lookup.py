"""nested_lookup — Two-level k1 -> k2 -> v associative lookup.

Extracted from the wave-2 candidate sweep where this task trained above
its modal baseline while staying below saturation on a 1M-param
transformer. See README for the full list of qualifying probes.
"""

from __future__ import annotations

import numpy as np

from .base import BOS, CONTENT_LO, MARKER, PAD, QUERY, SEP, TaskSpec


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    records = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    records[:, 0] = BOS
    return records, mask


def nested_lookup(rng, batch, spec: TaskSpec, n_pairs: int = 8):
    """Two dictionaries: a "redirect" table k1 -> k2 and a "value" table
    k2 -> v. Given query q=k1, emit v. Requires chaining two lookups.

    Layout: [BOS, k1_1, k2_1, ..., k1_n, k2_n, SEP1,
             k2_1, v_1, ..., k2_n, v_n, SEP2, q, ans]

    All three alphabets are disjoint content ranges.
    """
    T = spec.seq_len
    # budget: two tables of n_pairs each = 4n body + 2 SEPs + q + ans + BOS
    used = 1 + 4 * n_pairs + 2 + 2
    if used > T:
        raise ValueError("seq_len too short for nested_lookup")
    if spec.n_content < 3 * n_pairs:
        raise ValueError("vocab too small for nested_lookup")
    k1_lo = CONTENT_LO
    k2_lo = CONTENT_LO + n_pairs
    v_lo = CONTENT_LO + 2 * n_pairs
    recs, mask = _empty(batch, T)
    for b in range(batch):
        k1s = np.arange(n_pairs) + k1_lo
        k2s = (np.arange(n_pairs) + k2_lo).astype(np.int32)
        rng.shuffle(k2s)  # random redirect permutation
        vs = rng.integers(v_lo, v_lo + n_pairs, size=n_pairs, dtype=np.int32)
        # table 1: k1 -> k2 (in random visit order)
        order1 = rng.permutation(n_pairs)
        pos = 1
        for i in order1:
            recs[b, pos] = k1s[i]
            recs[b, pos + 1] = k2s[i]
            pos += 2
        recs[b, pos] = SEP
        pos += 1
        # table 2: k2 -> v (where k2 enumerated in a different shuffled order)
        # to look up by k2 we need each k2 value to appear; use the k2 alphabet.
        k2_all = (np.arange(n_pairs) + k2_lo).astype(np.int32)
        order2 = rng.permutation(n_pairs)
        # build a k2_value map that also covers values for the k2s that actually
        # appear as redirects; we map k2 (by value) -> vs
        k2_to_v = {int(k2s[i]): int(vs[i]) for i in range(n_pairs)}
        for i in order2:
            k2 = int(k2_all[i])
            recs[b, pos] = k2
            recs[b, pos + 1] = k2_to_v.get(k2, int(vs[0]))
            pos += 2
        recs[b, pos] = SEP
        pos += 1
        q_idx = int(rng.integers(0, n_pairs))
        recs[b, pos] = int(k1s[q_idx])
        recs[b, pos + 1] = int(vs[q_idx])
        mask[b, pos] = True
    return recs, mask
