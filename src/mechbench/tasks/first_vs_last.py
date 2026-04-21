"""first_vs_last — Bucketed distance between first and last occurrence of query.

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


def first_vs_last(rng, batch, spec: TaskSpec, body_len: int = 96, alpha: int = 6, n_buckets: int = 8):
    """Given query q, emit a bucketed distance between its FIRST and LAST
    occurrence in the body.

    Layout: [BOS, body..., SEP, q, distance_bucket]
    """
    T = spec.seq_len
    if body_len + 3 > T:
        raise ValueError("seq_len too short for first_vs_last")
    if spec.n_content < alpha + n_buckets:
        raise ValueError("vocab too small for first_vs_last")
    val_lo = CONTENT_LO
    bucket_lo = CONTENT_LO + alpha
    step = max(1, body_len // n_buckets)
    recs, mask = _empty(batch, T)
    for b in range(batch):
        body = rng.integers(0, alpha, size=body_len, dtype=np.int64)
        # pick a target distance bucket and plant query accordingly
        q_val = int(rng.integers(0, alpha))
        # strip all existing q_val from body
        body = np.where(body == q_val, (q_val + 1) % alpha, body)
        target_bucket = int(rng.integers(0, n_buckets))
        min_d = target_bucket * step
        max_d = min(body_len - 1, (target_bucket + 1) * step - 1)
        if max_d < min_d:
            max_d = min_d
        d = int(rng.integers(min_d, max_d + 1))
        i = int(rng.integers(0, body_len - d))
        j = i + d
        body[i] = q_val
        body[j] = q_val
        recs[b, 1 : 1 + body_len] = (val_lo + body).astype(np.int32)
        sep_pos = 1 + body_len
        recs[b, sep_pos] = SEP
        recs[b, sep_pos + 1] = val_lo + q_val
        recs[b, sep_pos + 2] = bucket_lo + target_bucket
        mask[b, sep_pos + 1] = True
    return recs, mask
