"""dual_hop_retrieve — Two independent queries, each 1 hop, emit both answers.

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


def dual_hop_retrieve(rng, batch, spec: TaskSpec, n_nodes: int = 12):
    T = spec.seq_len
    if spec.n_content < n_nodes:
        raise ValueError("vocab too small for dual_hop_retrieve")
    if 1 + 2 * n_nodes + 5 > T:
        raise ValueError("seq_len too short for dual_hop_retrieve")
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
        q1_idx, q2_idx = rng.choice(n_nodes, size=2, replace=False)
        q1 = int(nodes[int(q1_idx)])
        q2 = int(nodes[int(q2_idx)])
        recs[b, pos] = q1
        recs[b, pos + 1] = mapping[q1]
        recs[b, pos + 2] = q2
        recs[b, pos + 3] = mapping[q2]
        mask[b, pos] = True
        mask[b, pos + 2] = True
    return recs, mask
