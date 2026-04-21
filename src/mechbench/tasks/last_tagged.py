"""last_tagged — Emit the value of the LAST body pair with the query tag.

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


def last_tagged(rng, batch, spec: TaskSpec, body_pairs: int = 40, n_tags: int = 3):
    T = spec.seq_len
    body_len = 2 * body_pairs
    if body_len + 3 > T:
        raise ValueError("seq_len too short for last_tagged")
    if spec.n_content < n_tags + 4:
        raise ValueError("vocab too small for last_tagged")
    val_lo = CONTENT_LO
    tag_lo = CONTENT_LO + (spec.n_content - n_tags)
    val_alpha = tag_lo - val_lo
    if val_alpha < 2:
        raise ValueError("vocab partition too small")
    recs, mask = _empty(batch, T)
    for b in range(batch):
        q_tag = int(rng.integers(0, n_tags))
        tags = rng.integers(0, n_tags, size=body_pairs, dtype=np.int64)
        if q_tag not in tags:
            tags[rng.integers(0, body_pairs)] = q_tag
        vals = rng.integers(0, val_alpha, size=body_pairs, dtype=np.int64)
        pos = 1
        last_val = None
        for i in range(body_pairs):
            recs[b, pos] = tag_lo + int(tags[i])
            recs[b, pos + 1] = val_lo + int(vals[i])
            pos += 2
            if tags[i] == q_tag:
                last_val = int(vals[i])
        recs[b, pos] = SEP
        pos += 1
        recs[b, pos] = tag_lo + q_tag
        recs[b, pos + 1] = val_lo + int(last_val)
        mask[b, pos] = True
    return recs, mask
