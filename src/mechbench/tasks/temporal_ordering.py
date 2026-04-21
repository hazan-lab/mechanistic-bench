"""temporal_ordering — Binary: did token A appear before token B in the body?

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


def temporal_ordering(rng, batch, spec: TaskSpec, body_len: int = 96, alpha: int = 8):
    """Binary: A occurs before B in the body (yes/no). Body contains
    exactly one A and one B; other positions drawn from disjoint filler
    alphabet.

    Layout: [BOS, body..., SEP, A, B, yes/no]
    """
    T = spec.seq_len
    if body_len + 4 > T:
        raise ValueError("seq_len too short for temporal_ordering")
    if spec.n_content < alpha + 2:
        raise ValueError("vocab too small for temporal_ordering")
    val_lo = CONTENT_LO
    yes_tok = CONTENT_LO + alpha
    no_tok = yes_tok + 1
    recs, mask = _empty(batch, T)
    for b in range(batch):
        A, B = rng.choice(alpha, size=2, replace=False)
        filler_pool = [v for v in range(alpha) if v != A and v != B]
        body = np.array(
            [int(rng.choice(np.array(filler_pool))) for _ in range(body_len)],
            dtype=np.int32,
        )
        pA = int(rng.integers(1, body_len - 1))
        if rng.random() < 0.5:
            pB = int(rng.integers(pA + 1, body_len))
        else:
            pB = int(rng.integers(0, pA))
        before = pA < pB  # A before B?
        body[pA] = int(A)
        body[pB] = int(B)
        recs[b, 1 : 1 + body_len] = val_lo + body
        sep_pos = 1 + body_len
        recs[b, sep_pos] = SEP
        recs[b, sep_pos + 1] = val_lo + int(A)
        recs[b, sep_pos + 2] = val_lo + int(B)
        recs[b, sep_pos + 3] = yes_tok if before else no_tok
        mask[b, sep_pos + 2] = True
    return recs, mask
