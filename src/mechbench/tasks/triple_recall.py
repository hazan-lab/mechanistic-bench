"""triple_recall — 3 parallel queries against one key-value table.

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


def triple_recall(rng, batch, spec: TaskSpec, n_pairs: int = 16, n_queries: int = 3):
    """Batch recall with just 3 queries (between single recall and full
    batch_recall).

    Layout: [BOS, k_1, v_1, ..., k_n, v_n, SEP, q_1, v_1, ..., q_3, v_3]
    """
    T = spec.seq_len
    key_alpha = spec.n_content // 2
    val_alpha = spec.n_content - key_alpha
    if n_pairs > key_alpha:
        raise ValueError("n_pairs > key alphabet")
    used = 1 + 2 * n_pairs + 1 + 2 * n_queries
    if used > T:
        raise ValueError("seq_len too short for triple_recall")
    key_lo = CONTENT_LO
    val_lo = CONTENT_LO + key_alpha
    recs, mask = _empty(batch, T)
    for b in range(batch):
        key_pool = rng.permutation(key_alpha)[:n_pairs] + key_lo
        vals = rng.integers(val_lo, val_lo + val_alpha, size=n_pairs, dtype=np.int32)
        pos = 1
        for i in range(n_pairs):
            recs[b, pos] = key_pool[i]
            recs[b, pos + 1] = vals[i]
            pos += 2
        recs[b, pos] = SEP
        pos += 1
        q_idx = rng.integers(0, n_pairs, size=n_queries)
        for j in range(n_queries):
            recs[b, pos] = key_pool[q_idx[j]]
            recs[b, pos + 1] = vals[q_idx[j]]
            mask[b, pos] = True
            pos += 2
    return recs, mask
