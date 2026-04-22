"""conditional_recall — Branch on presence of control token, then recall.

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


def conditional_recall(rng, batch, spec: TaskSpec, n_pairs: int = 10):
    """If a control token ``c_hi`` is present in the body, emit value of
    key ``k_hi``; else emit value of key ``k_lo``.

    Layout: [BOS, k1, v1, ..., kn, vn, control?, SEP, q_hi, q_lo, ans]

    Control token appears before SEP with p=0.5; if present, answer is
    v(k_hi); else v(k_lo). The model must (a) detect the control token,
    (b) pick the correct key to look up, and (c) retrieve.
    """
    T = spec.seq_len
    key_alpha = n_pairs
    val_alpha = spec.n_content - key_alpha - 1
    if val_alpha < 2 or key_alpha < 4:
        raise ValueError("vocab too small for conditional_recall")
    used = 1 + 2 * n_pairs + 1 + 4
    if used > T:
        raise ValueError("seq_len too short for conditional_recall")
    key_lo = CONTENT_LO
    val_lo = CONTENT_LO + key_alpha
    ctrl_tok = spec.vocab_size - 1
    recs, mask = _empty(batch, T)
    for b in range(batch):
        keys = rng.permutation(key_alpha)[:n_pairs] + key_lo
        vals = rng.integers(val_lo, val_lo + val_alpha, size=n_pairs, dtype=np.int32)
        pos = 1
        for i in range(n_pairs):
            recs[b, pos] = keys[i]
            recs[b, pos + 1] = vals[i]
            pos += 2
        present = int(rng.integers(0, 2))
        if present:
            recs[b, pos] = ctrl_tok
            pos += 1
        recs[b, pos] = SEP
        sep_pos = pos
        pos += 1
        q_hi_idx, q_lo_idx = rng.choice(n_pairs, size=2, replace=False)
        recs[b, pos] = keys[q_hi_idx]
        recs[b, pos + 1] = keys[q_lo_idx]
        recs[b, pos + 2] = vals[q_hi_idx] if present else vals[q_lo_idx]
        mask[b, pos + 1] = True
    return recs, mask
