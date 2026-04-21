"""two_hop — Fixed 2-hop retrieval on functional graph (n_nodes=14).

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


def two_hop(rng, batch, spec: TaskSpec, n_nodes: int = 14):
    """Fixed 2-hop retrieval (like deep_hop with k=2 but larger graph).

    Layout: [BOS, src_1, dst_1, ..., src_n, dst_n, SEP, start, MARKER, MARKER, ans]
    """
    T = spec.seq_len
    if spec.n_content < n_nodes:
        raise ValueError("vocab too small for two_hop")
    reserve = 5
    if 1 + 2 * n_nodes + reserve > T:
        raise ValueError("seq_len too short for two_hop")
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
        start_idx = int(rng.integers(0, n_nodes))
        start = int(nodes[start_idx])
        recs[b, pos] = start
        recs[b, pos + 1] = MARKER
        recs[b, pos + 2] = MARKER
        pos += 3
        mapping = {int(nodes[i]): int(dsts[i]) for i in range(n_nodes)}
        cur = start
        cur = mapping[cur]
        cur = mapping[cur]
        recs[b, pos] = cur
        mask[b, pos - 1] = True
    return recs, mask
