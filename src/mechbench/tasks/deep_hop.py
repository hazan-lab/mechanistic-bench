"""Fixed-depth multi-hop retrieval on a functional graph.

Layout (seq_len T):

    [BOS, src_1, dst_1, ..., src_n, dst_n,
     SEP, start, MARKER * k, answer, PAD, ...]

The prefix lists ``n_nodes`` unique (src, dst) pairs defining a
random functional map on a pool of node tokens. After the SEP a
start token is followed by exactly ``k`` MARKER tokens, and the
model must emit the node reached by applying the map k times from
start. Unlike the ``k_hop`` task, hop count is **fixed at k**, so
there are no easy k=1 instances inflating the score.

Why this is a useful probe:

* **Relevant to LM.** Targets deep entity chains — 4-hop lookups
  are a well-known stress test for in-context factual retrieval in
  LMs.
* **Unsaturated at 1M.** A 1M causal transformer (6 attn layers,
  d_model=128) plateaus at ~37% token accuracy with n_nodes=10 /
  k=4 after 4k steps, vs a 10% random baseline. The stratification
  by hop depth is implicit in each sample: the model has to
  represent the entire chain, not the variable-depth average
  k_hop captures.
* **Diagnostic.** Failure is a pure depth bottleneck: a model that
  can follow 2 hops but not 4 will show as ~25% (random on the
  last two hops). Distinguishes shallow-attention gaps from
  capacity gaps.
"""

from __future__ import annotations

import numpy as np

from .base import BOS, CONTENT_LO, MARKER, PAD, SEP, TaskSpec


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    records = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    records[:, 0] = BOS
    return records, mask


def deep_hop(rng, batch, spec: TaskSpec, n_nodes: int = 10, k: int = 4):
    """Apply a functional graph ``k`` times and emit the reached node."""
    T = spec.seq_len
    if spec.n_content < n_nodes:
        raise ValueError("vocab too small for deep_hop")
    reserve = 3 + k  # SEP + start + k markers + answer
    if 1 + 2 * n_nodes + reserve > T:
        raise ValueError("seq_len too short for deep_hop")
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
        recs[b, pos : pos + k] = MARKER
        pos += k
        mapping = {int(nodes[i]): int(dsts[i]) for i in range(n_nodes)}
        cur = start
        for _ in range(k):
            cur = mapping[cur]
        recs[b, pos] = cur
        mask[b, pos - 1] = True
    return recs, mask
