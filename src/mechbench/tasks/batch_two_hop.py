"""batch_two_hop — 3 parallel queries, each requiring exactly 2 hops.

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


def batch_two_hop(rng, batch, spec: TaskSpec, n_nodes: int = 10, n_queries: int = 3):
    """Functional graph on n_nodes. Each query is answered with the node
    reached after applying the map exactly twice.

    Layout: [BOS, src_1, dst_1, ..., src_n, dst_n, SEP,
             q_1, ans_1, q_2, ans_2, q_3, ans_3]
    """
    T = spec.seq_len
    if spec.n_content < n_nodes:
        raise ValueError("vocab too small for batch_two_hop")
    if 1 + 2 * n_nodes + 1 + 2 * n_queries > T:
        raise ValueError("seq_len too short for batch_two_hop")
    node_lo = CONTENT_LO
    recs, mask = _empty(batch, T)
    for b in range(batch):
        nodes = np.arange(n_nodes) + node_lo
        dsts = rng.choice(nodes, size=n_nodes, replace=True)
        order = rng.permutation(n_nodes)
        pos = 1
        for i in order:
            recs[b, pos] = nodes[i]
            recs[b, pos + 1] = dsts[i]
            pos += 2
        recs[b, pos] = SEP
        pos += 1
        mapping = {int(nodes[i]): int(dsts[i]) for i in range(n_nodes)}
        q_idx = rng.integers(0, n_nodes, size=n_queries)
        for j in range(n_queries):
            q = int(nodes[int(q_idx[j])])
            ans = mapping[mapping[q]]
            recs[b, pos] = q
            recs[b, pos + 1] = ans
            mask[b, pos] = True
            pos += 2
    return recs, mask
