"""three_hop — Fixed 3-hop retrieval on a functional graph (n_nodes=12).

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


def three_hop(rng, batch, spec: TaskSpec, n_nodes: int = 12):
    T = spec.seq_len
    k = 3
    if spec.n_content < n_nodes:
        raise ValueError("vocab too small for three_hop")
    if 1 + 2 * n_nodes + 3 + k > T:
        raise ValueError("seq_len too short for three_hop")
    recs, mask = _empty(batch, T)
    pool = np.arange(CONTENT_LO, CONTENT_LO + n_nodes, dtype=np.int32)
    for b in range(batch):
        nodes = pool.copy()
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
        start = int(nodes[int(rng.integers(0, n_nodes))])
        recs[b, pos] = start
        recs[b, pos + 1 : pos + 1 + k] = MARKER
        pos += 1 + k
        mapping = {int(nodes[i]): int(dsts[i]) for i in range(n_nodes)}
        cur = start
        for _ in range(k):
            cur = mapping[cur]
        recs[b, pos] = cur
        mask[b, pos - 1] = True
    return recs, mask
