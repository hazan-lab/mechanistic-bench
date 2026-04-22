"""k-hop retrieval over a functional graph.

Layout (seq_len T):

    [BOS, src_1, dst_1, src_2, dst_2, ..., src_n, dst_n,
     SEP, start, MARKER * k, answer, PAD, ...]

The (src_i, dst_i) pairs define a random functional graph on ``n_nodes``
node tokens (each src appears exactly once; dst is sampled with
replacement from the node pool). After the SEP the stream gives a
``start`` node and encodes a hop count ``k`` in ``1..max_hops`` as a
run of ``MARKER`` tokens. The model must emit the node reached by
following the map ``k`` times from ``start``.

Why this is a useful probe:

* **Relevant to LM.** Directly isolates multi-hop entity chaining,
  which is a persistent bottleneck in factual-LM evaluations.
* **Unsaturated at 1M.** At n_nodes=10, max_hops=3 a 1M transformer
  plateaus near ~28% token accuracy (vs 10% random) within a few
  thousand steps — well below saturation, but clearly above chance,
  so the signal is diagnostic of architectural depth.
* **Diagnostic.** Solving k hops requires effectively ``k`` sequential
  lookups; shallow / limited-state models should stratify by hop
  count, exposing genuine architectural gaps rather than a binary
  pass/fail.
"""

from __future__ import annotations

import numpy as np

from .base import BOS, CONTENT_LO, MARKER, PAD, SEP, TaskSpec


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    records = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    records[:, 0] = BOS
    return records, mask


def k_hop(rng, batch, spec: TaskSpec, n_nodes: int = 10, max_hops: int = 3):
    """k-hop retrieval on a random functional graph."""
    T = spec.seq_len
    if spec.n_content < n_nodes:
        raise ValueError("vocab too small for k_hop")
    reserve = 3 + max_hops  # SEP + start + markers + answer
    if 1 + 2 * n_nodes + reserve > T:
        raise ValueError("seq_len too short for k_hop")
    recs, mask = _empty(batch, T)
    node_pool = np.arange(CONTENT_LO, CONTENT_LO + n_nodes, dtype=np.int32)
    for b in range(batch):
        nodes = node_pool.copy()
        rng.shuffle(nodes)
        dsts = rng.choice(nodes, size=n_nodes, replace=True)
        order = rng.permutation(n_nodes)
        pos = 1
        for i in order:
            recs[b, pos] = nodes[i]
            recs[b, pos + 1] = dsts[i]
            pos += 2
        recs[b, pos] = SEP
        pos += 1
        start_idx = int(rng.integers(0, n_nodes))
        start = int(nodes[start_idx])
        recs[b, pos] = start
        pos += 1
        k = int(rng.integers(1, max_hops + 1))
        recs[b, pos : pos + k] = MARKER
        pos += k
        mapping = {int(nodes[i]): int(dsts[i]) for i in range(n_nodes)}
        cur = start
        for _ in range(k):
            cur = mapping[cur]
        recs[b, pos] = cur
        mask[b, pos - 1] = True
    return recs, mask
