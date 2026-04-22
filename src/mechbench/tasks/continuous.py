"""Continuous-modality mechanistic tasks.

These treat a token stream as a discretised continuous signal (level tokens
drawn from a small ordinal alphabet) and isolate primitives that matter
for time-series and audio-like modalities:

    delayed_echo        — interleaved (sample, echo) pairs where echo_i =
                          sample_{i-lag}. The model must hold a fixed-depth
                          queue, which is the defining operation of a
                          state-space layer.

    piecewise_denoise   — noisy piecewise-constant signal denoised to its
                          underlying level. Probes windowed-majority
                          aggregation (local smoothing).

    nearest_key         — ordinal associative recall: key/value pairs with
                          integer-valued keys; the query may be a key that
                          was not observed, and the target is the value
                          whose key is *closest* in ordinal distance.
                          Tests continuous-valued (non-exact-match)
                          associative recall.
"""

from __future__ import annotations

import numpy as np

from .base import CONTENT_LO, PAD, SEP, TaskSpec


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    from .base import BOS
    records = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    records[:, 0] = BOS
    return records, mask


def delayed_echo(rng, batch, spec: TaskSpec, lag: int = 3):
    """Interleaved stream (s_1, e_1, s_2, e_2, ...) with e_i = s_{i-lag}.

    Positions i <= lag have no well-defined echo and are masked out so only
    positions where the model must recall something ``lag`` steps back
    count toward the loss. Alphabet is small (default 8 levels) so the
    task is learnable but not trivial — a model with state depth < lag
    cannot solve it.
    """
    T = spec.seq_len
    if lag < 1:
        raise ValueError("delayed_echo requires lag >= 1")
    n_pairs = (T - 1) // 2
    if n_pairs <= lag:
        raise ValueError("seq_len too short for delayed_echo with given lag")
    recs, mask = _empty(batch, T)
    alpha = min(8, spec.n_content)
    body = rng.integers(CONTENT_LO, CONTENT_LO + alpha, size=(batch, n_pairs), dtype=np.int32)
    # sample positions are at 1 + 2t, echoes at 1 + 2t + 1.
    for t in range(n_pairs):
        s_pos = 1 + 2 * t
        e_pos = s_pos + 1
        if e_pos > T:
            break
        recs[:, s_pos] = body[:, t]
        if t >= lag:
            recs[:, e_pos] = body[:, t - lag]
            # label at input position s_pos is records[s_pos + 1] = echo
            mask[:, s_pos] = True
        else:
            # pre-warm: emit an arbitrary in-alphabet token, masked out.
            recs[:, e_pos] = CONTENT_LO
    return recs, mask


def piecewise_denoise(rng, batch, spec: TaskSpec, noise_prob: float = 0.15, min_seg: int = 5):
    """Denoise a noisy piecewise-constant signal.

    Body is a concatenation of constant segments, each of length >= min_seg,
    with each position independently corrupted to a uniform-random level
    with probability ``noise_prob``. After SEP, the target is the clean
    per-position level. The optimal strategy is windowed majority within
    each segment, which requires local aggregation of state.
    """
    T = spec.seq_len
    body_len = (T - 2) // 2
    if body_len < 2 * min_seg:
        raise ValueError("seq_len too short for piecewise_denoise")
    recs, mask = _empty(batch, T)
    alpha = 4
    for b in range(batch):
        # partition body_len into segments of length >= min_seg
        segs: list[int] = []
        remaining = body_len
        while remaining >= 2 * min_seg:
            ln = int(rng.integers(min_seg, remaining - min_seg + 1))
            # keep segments reasonably short on average
            ln = min(ln, min_seg + 6)
            segs.append(ln)
            remaining -= ln
        segs.append(remaining)
        clean = np.empty(body_len, dtype=np.int32)
        cursor = 0
        prev_level = -1
        for ln in segs:
            lev = int(rng.integers(0, alpha))
            # avoid two adjacent segments sharing a level (keeps boundaries well-defined)
            if lev == prev_level:
                lev = (lev + 1) % alpha
            clean[cursor : cursor + ln] = lev
            cursor += ln
            prev_level = lev
        noisy = clean.copy()
        flip = rng.random(body_len) < noise_prob
        if flip.any():
            noisy[flip] = rng.integers(0, alpha, size=int(flip.sum()), dtype=np.int32)
        recs[b, 1 : 1 + body_len] = CONTENT_LO + noisy
        sep_pos = 1 + body_len
        recs[b, sep_pos] = SEP
        out_len = min(body_len, T - sep_pos - 1)
        recs[b, sep_pos + 1 : sep_pos + 1 + out_len] = CONTENT_LO + clean[:out_len]
        mask[b, sep_pos : sep_pos + out_len] = True
    return recs, mask


def nearest_key(rng, batch, spec: TaskSpec):
    """Ordinal associative recall.

    Layout: k_1, v_1, k_2, v_2, ..., SEP, query_key, target_value.
    Keys are drawn without replacement from an ordinal range of size
    ``n_keys``; values come from a disjoint value range. The query is a
    uniform sample from the full key range and may not appear among the
    observed keys; the target is the value whose key minimises |k - q|
    (ties broken to the lower index). Probes whether the model can learn
    an ordinal similarity rather than exact-match key comparison.
    """
    T = spec.seq_len
    n_keys = min(16, spec.n_content // 2)
    if n_keys < 4:
        raise ValueError("vocab_size too small for nearest_key")
    val_lo = CONTENT_LO + n_keys
    val_hi = spec.vocab_size
    if val_hi - val_lo < 2:
        raise ValueError("not enough room for nearest_key value alphabet")
    max_pairs_by_seq = max(2, (T - 4) // 2)
    n_pairs = min(n_keys, max_pairs_by_seq)
    recs, mask = _empty(batch, T)
    for b in range(batch):
        keys = rng.choice(np.arange(n_keys), size=n_pairs, replace=False)
        vals = rng.integers(val_lo, val_hi, size=n_pairs, dtype=np.int32)
        flat = np.empty(2 * n_pairs, dtype=np.int32)
        flat[0::2] = CONTENT_LO + keys.astype(np.int32)
        flat[1::2] = vals
        recs[b, 1 : 1 + 2 * n_pairs] = flat
        sep_pos = 1 + 2 * n_pairs
        recs[b, sep_pos] = SEP
        q = int(rng.integers(0, n_keys))
        # nearest present key (argmin breaks ties by lowest index)
        dists = np.abs(keys - q)
        near_idx = int(np.argmin(dists))
        recs[b, sep_pos + 1] = CONTENT_LO + q
        recs[b, sep_pos + 2] = vals[near_idx]
        mask[b, sep_pos + 1] = True
    return recs, mask
