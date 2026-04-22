"""substring_locate — Find a planted 3-token substring; emit bucketed start-position.

Extracted from the wave-5 candidate sweep (algorithmic/sequence
primitives): this task trains above its modal baseline yet stays below
saturation on a 1M-param transformer.
"""

from __future__ import annotations

import numpy as np

from .base import BOS, CONTENT_LO, MARKER, PAD, QUERY, SEP, TaskSpec


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    recs = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    recs[:, 0] = BOS
    return recs, mask


def substring_locate(rng, batch, spec: TaskSpec, body_len: int = 64, alpha: int = 8, n_buckets: int = 8):
    """Planted 3-token substring occurs exactly once in body. Emit bucketed
    start-position. Substring tokens sampled to be sufficiently rare as a
    triple in the surrounding random body.

    Layout: [BOS, body..., SEP, s1, s2, s3, bucket]
    """
    T = spec.seq_len
    if body_len + 5 > T:
        raise ValueError("seq_len too short for substring_locate")
    if spec.n_content < alpha + n_buckets:
        raise ValueError("vocab too small for substring_locate")
    val_lo = CONTENT_LO
    bucket_lo = val_lo + alpha
    step = max(1, (body_len - 3) // n_buckets)
    recs, mask = _empty(batch, T)
    for b in range(batch):
        target_bucket = int(rng.integers(0, n_buckets))
        start = int(rng.integers(target_bucket * step, min((target_bucket + 1) * step, body_len - 2)))
        body = rng.integers(0, alpha, size=body_len, dtype=np.int32)
        s1, s2, s3 = rng.integers(0, alpha, size=3, dtype=np.int32)
        # remove any accidental occurrence of (s1, s2, s3) elsewhere
        for i in range(body_len - 2):
            if body[i] == s1 and body[i + 1] == s2 and body[i + 2] == s3:
                body[i] = (int(body[i]) + 1) % alpha
        body[start] = s1
        body[start + 1] = s2
        body[start + 2] = s3
        recs[b, 1 : 1 + body_len] = val_lo + body
        sep_pos = 1 + body_len
        recs[b, sep_pos] = SEP
        recs[b, sep_pos + 1] = val_lo + int(s1)
        recs[b, sep_pos + 2] = val_lo + int(s2)
        recs[b, sep_pos + 3] = val_lo + int(s3)
        recs[b, sep_pos + 4] = bucket_lo + target_bucket
        mask[b, sep_pos + 3] = True
    return recs, mask
