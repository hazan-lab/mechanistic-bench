"""mode_tagged — Mode of values sharing the queried tag.

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


def mode_tagged(rng, batch, spec: TaskSpec, body_alpha: int = 5, n_tags: int = 3, body_pairs: int = 40):
    """For each of ``body_pairs`` positions the body contains a (tag, value)
    pair. Given a query tag, emit the modal value among positions with
    that tag.

    Layout: [BOS, t_1, v_1, ..., t_L, v_L, SEP, q_tag, mode_value]
    """
    T = spec.seq_len
    body_len = 2 * body_pairs
    if body_len + 3 > T:
        raise ValueError("seq_len too short for mode_tagged")
    if spec.n_content < body_alpha + n_tags:
        raise ValueError("vocab too small for mode_tagged")
    val_lo = CONTENT_LO
    tag_lo = CONTENT_LO + body_alpha
    recs, mask = _empty(batch, T)
    for b in range(batch):
        tags = rng.integers(0, n_tags, size=body_pairs, dtype=np.int64)
        # pick the query tag, ensure it appears enough times that there is a unique mode
        while True:
            q_tag = int(rng.integers(0, n_tags))
            sel_idx = np.where(tags == q_tag)[0]
            if len(sel_idx) >= 3:
                break
        vals = rng.integers(0, body_alpha, size=body_pairs, dtype=np.int64)
        # enforce a unique mode among the selected values
        sel_vals = vals[sel_idx].copy()
        target_mode = int(rng.integers(0, body_alpha))
        # make at least half of them equal the target_mode
        k = (len(sel_idx) // 2) + 1
        sel_vals[:k] = target_mode
        rng.shuffle(sel_vals)
        vals[sel_idx] = sel_vals
        pos = 1
        for i in range(body_pairs):
            recs[b, pos] = tag_lo + int(tags[i])
            recs[b, pos + 1] = val_lo + int(vals[i])
            pos += 2
        recs[b, pos] = SEP
        pos += 1
        recs[b, pos] = tag_lo + q_tag
        recs[b, pos + 1] = val_lo + target_mode
        mask[b, pos] = True
    return recs, mask
