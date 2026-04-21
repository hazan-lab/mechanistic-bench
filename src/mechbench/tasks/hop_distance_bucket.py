"""hop_distance_bucket — Bucketed distance (in hops) from u to v in a functional graph.

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


def hop_distance_bucket(rng, batch, spec: TaskSpec, n_nodes: int = 10, max_dist: int = 6):
    """Given (src, dst) edges + a query (u, v), emit the number of hops from
    u to v (or an unreachable marker), bucketed into max_dist bins.

    Layout: [BOS, src_1, dst_1, ..., src_n, dst_n, SEP, u, v, dist_bucket]
    """
    T = spec.seq_len
    if spec.n_content < n_nodes + max_dist + 1:
        raise ValueError("vocab too small for hop_distance_bucket")
    if 1 + 2 * n_nodes + 4 > T:
        raise ValueError("seq_len too short for hop_distance_bucket")
    node_lo = CONTENT_LO
    dist_lo = CONTENT_LO + n_nodes
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
        u = int(nodes[rng.integers(0, n_nodes)])
        # sample target distance uniformly and trace
        target = int(rng.integers(0, max_dist))
        cur = u
        for _ in range(target):
            cur = mapping[cur]
        v = cur
        # compute actual distance for label (cap at max_dist-1)
        d = 0
        cur2 = u
        visited = {u: 0}
        while cur2 != v and d < max_dist - 1:
            cur2 = mapping[cur2]
            d += 1
            if cur2 in visited:
                break
            visited[cur2] = d
        recs[b, pos] = u
        recs[b, pos + 1] = v
        recs[b, pos + 2] = dist_lo + d
        mask[b, pos + 1] = True
    return recs, mask
