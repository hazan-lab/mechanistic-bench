"""Batch associative recall: answer multiple queries from one key-value table.

Layout (seq_len T):

    [BOS, k_1, v_1, k_2, v_2, ..., k_n, v_n,
     SEP, q_1, v_1, q_2, v_2, ..., q_m, v_m, PAD, ...]

The prefix lists ``n_pairs`` unique (key, value) pairs drawn from two
disjoint content ranges (first half keys, second half values). After
the SEP the stream interleaves a sequence of ``n_queries`` keys with
their associated values; the model must predict each value from the
preceding table and query. Previous answers do not help because each
query picks an independent row.

Why this is a useful probe:

* **Relevant to LM.** Isolates parallel retrieval — the primitive
  behind in-context-learning-style lookups in language models. Single
  associative recall already exists in the suite; batching queries
  exposes whether a model can share one attention head's retrieval
  circuit across multiple slots.
* **Unsaturated at 1M.** With ``n_pairs=16`` and ``n_queries=5`` a 1M
  causal transformer plateaus near **14.5% token accuracy / 0.1%
  sequence accuracy** after 4k steps — 8x random per-token but still
  far from solving any single sample outright.
* **Diagnostic.** Per-query accuracy decomposes the error: if a model
  solves the first query but fails subsequent ones, that points to a
  head that cannot generalise its lookup circuit across positions,
  which is a real architectural signature.
"""

from __future__ import annotations

import numpy as np

from .base import BOS, CONTENT_LO, PAD, SEP, TaskSpec


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    records = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    records[:, 0] = BOS
    return records, mask


def batch_recall(rng, batch, spec: TaskSpec, n_pairs: int = 16, n_queries: int = 5):
    """Retrieve ``n_queries`` values in one pass from a shared key-value table."""
    T = spec.seq_len
    key_alpha = spec.n_content // 2
    val_alpha = spec.n_content - key_alpha
    if n_pairs > key_alpha:
        raise ValueError("n_pairs > key alphabet")
    if val_alpha < 2 or key_alpha < 4:
        raise ValueError("vocab too small for batch_recall")
    used = 1 + 2 * n_pairs + 1 + 2 * n_queries
    if used > T:
        raise ValueError("seq_len too short for batch_recall")
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
