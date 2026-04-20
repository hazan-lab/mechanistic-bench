"""Synthetic mechanistic tasks.

Each generator returns ``(records, loss_mask)`` arrays of shapes
``(batch, seq_len+1)`` and ``(batch, seq_len)`` respectively. Records are int32
token ids; loss_mask is bool, True at positions we want to evaluate.

All tasks use the shared vocabulary convention from ``base.py``:
    PAD=0, BOS=1, SEP=2, QUERY=3, MARKER=4, content=[5, vocab_size).
"""

from __future__ import annotations

import numpy as np

from .base import (
    BOS,
    CONTENT_LO,
    MARKER,
    PAD,
    QUERY,
    SEP,
    TaskSpec,
    sample_content,
)


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    records = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    records[:, 0] = BOS
    return records, mask


# ---------------------------------------------------------------------------
# Retrieval-style tasks (favour attention)
# ---------------------------------------------------------------------------


def copy(rng, batch, spec: TaskSpec):
    T = spec.seq_len
    k = T // 3
    recs, mask = _empty(batch, T)
    payload = sample_content(rng, spec, (batch, k))
    recs[:, 1 : 1 + k] = payload
    recs[:, 1 + k] = SEP
    recs[:, 2 + k : 2 + 2 * k] = payload
    # labels live at targets[t] = records[t+1]; we want to predict positions
    # that are the "copy region". label for position p is records[p+1].
    # We evaluate on positions k+1..2k (0-indexed input positions), whose
    # targets are records[k+2..2k+2], i.e. the copy region.
    mask[:, k + 1 : 2 * k + 1] = True
    return recs, mask


def copy_offset(rng, batch, spec: TaskSpec, offset: int = 5):
    """Copy payload starting `offset` tokens after the SEP."""
    T = spec.seq_len
    k = (T - offset) // 3
    recs, mask = _empty(batch, T)
    payload = sample_content(rng, spec, (batch, k))
    recs[:, 1 : 1 + k] = payload
    recs[:, 1 + k] = SEP
    recs[:, 2 + k : 2 + k + offset] = PAD  # already PAD
    start = 2 + k + offset
    recs[:, start : start + k] = payload
    mask[:, start - 1 : start - 1 + k] = True
    return recs, mask


def reverse_copy(rng, batch, spec: TaskSpec):
    T = spec.seq_len
    k = T // 3
    recs, mask = _empty(batch, T)
    payload = sample_content(rng, spec, (batch, k))
    recs[:, 1 : 1 + k] = payload
    recs[:, 1 + k] = SEP
    recs[:, 2 + k : 2 + 2 * k] = payload[:, ::-1]
    mask[:, k + 1 : 2 * k + 1] = True
    return recs, mask


def induction(rng, batch, spec: TaskSpec):
    """Place pattern "A B" twice; predict B after the second A."""
    T = spec.seq_len
    recs, mask = _empty(batch, T)
    # fill filler with content tokens
    filler = sample_content(rng, spec, (batch, T))
    recs[:, 1:T] = filler[:, 1:T]
    pos1 = T // 4
    pos2 = 3 * T // 4
    A = sample_content(rng, spec, (batch,))
    B = sample_content(rng, spec, (batch,))
    # ensure A != B so we can tell if the model memorised vs matched
    same = A == B
    while np.any(same):
        B = np.where(same, sample_content(rng, spec, (batch,)), B)
        same = A == B
    recs[np.arange(batch), pos1] = A
    recs[np.arange(batch), pos1 + 1] = B
    recs[np.arange(batch), pos2] = A
    recs[np.arange(batch), pos2 + 1] = B
    mask[:, pos2] = True  # label at input position pos2 is records[pos2+1] = B
    return recs, mask


def induction_gap(rng, batch, spec: TaskSpec, gap: int = 1):
    """A ? B ... A ? [B]. 1-token gap between trigger and target."""
    T = spec.seq_len
    recs, mask = _empty(batch, T)
    filler = sample_content(rng, spec, (batch, T))
    recs[:, 1:T] = filler[:, 1:T]
    pos1 = T // 4
    pos2 = 3 * T // 4
    A = sample_content(rng, spec, (batch,))
    B = sample_content(rng, spec, (batch,))
    same = A == B
    while np.any(same):
        B = np.where(same, sample_content(rng, spec, (batch,)), B)
        same = A == B
    idx = np.arange(batch)
    recs[idx, pos1] = A
    # gap tokens in between
    recs[idx, pos1 + 1 + gap] = B
    recs[idx, pos2] = A
    recs[idx, pos2 + 1 + gap] = B
    mask[idx, pos2 + gap] = True
    return recs, mask


def multi_induction(rng, batch, spec: TaskSpec):
    """3-token trigger A B C repeated as prefix A B, predict C."""
    T = spec.seq_len
    recs, mask = _empty(batch, T)
    filler = sample_content(rng, spec, (batch, T))
    recs[:, 1:T] = filler[:, 1:T]
    pos1 = T // 4
    pos2 = 3 * T // 4
    A = sample_content(rng, spec, (batch,))
    B = sample_content(rng, spec, (batch,))
    C = sample_content(rng, spec, (batch,))
    idx = np.arange(batch)
    recs[idx, pos1] = A
    recs[idx, pos1 + 1] = B
    recs[idx, pos1 + 2] = C
    recs[idx, pos2] = A
    recs[idx, pos2 + 1] = B
    recs[idx, pos2 + 2] = C
    mask[idx, pos2 + 1] = True  # predict C after seeing A B
    return recs, mask


def associative(rng, batch, spec: TaskSpec):
    """Key-value pairs followed by SEP then a key; predict its value."""
    T = spec.seq_len
    n_pairs = max(4, T // 6)
    recs, mask = _empty(batch, T)
    if 1 + 2 * n_pairs + 3 >= T:
        n_pairs = max(2, (T - 4) // 2 // 2)
    # split vocabulary: keys come from first half, values from second half
    half = spec.n_content // 2
    key_lo, key_hi = CONTENT_LO, CONTENT_LO + half
    val_lo, val_hi = CONTENT_LO + half, spec.vocab_size
    # sample unique keys per sample
    idx = np.arange(batch)
    for b in range(batch):
        keys = rng.choice(np.arange(key_lo, key_hi), size=n_pairs, replace=False)
        vals = rng.integers(val_lo, val_hi, size=n_pairs, dtype=np.int32)
        flat = np.empty(2 * n_pairs, dtype=np.int32)
        flat[0::2] = keys
        flat[1::2] = vals
        recs[b, 1 : 1 + 2 * n_pairs] = flat
        sep_pos = 1 + 2 * n_pairs
        recs[b, sep_pos] = SEP
        q_idx = int(rng.integers(0, n_pairs))
        recs[b, sep_pos + 1] = keys[q_idx]
        recs[b, sep_pos + 2] = vals[q_idx]
        mask[b, sep_pos + 1] = True
    return recs, mask


def short_associative(rng, batch, spec: TaskSpec):
    """Same as associative but fixed n_pairs=6."""
    old = spec.seq_len
    T = old
    n_pairs = 6
    recs, mask = _empty(batch, T)
    half = spec.n_content // 2
    key_lo, key_hi = CONTENT_LO, CONTENT_LO + half
    val_lo, val_hi = CONTENT_LO + half, spec.vocab_size
    need = 1 + 2 * n_pairs + 3
    if need >= T:
        n_pairs = max(2, (T - 4) // 2 // 2)
    for b in range(batch):
        keys = rng.choice(np.arange(key_lo, key_hi), size=n_pairs, replace=False)
        vals = rng.integers(val_lo, val_hi, size=n_pairs, dtype=np.int32)
        flat = np.empty(2 * n_pairs, dtype=np.int32)
        flat[0::2] = keys
        flat[1::2] = vals
        recs[b, 1 : 1 + 2 * n_pairs] = flat
        sep_pos = 1 + 2 * n_pairs
        recs[b, sep_pos] = SEP
        q_idx = int(rng.integers(0, n_pairs))
        recs[b, sep_pos + 1] = keys[q_idx]
        recs[b, sep_pos + 2] = vals[q_idx]
        mask[b, sep_pos + 1] = True
    return recs, mask


def selective_copy(rng, batch, spec: TaskSpec):
    """Copy tokens preceded by MARKER into the output region."""
    T = spec.seq_len
    src_len = T // 2
    recs, mask = _empty(batch, T)
    # reserve space for up to src_len/3 markers; we'll truncate output if longer
    max_out = src_len // 3
    content = sample_content(rng, spec, (batch, src_len))
    # place markers ~1/3 of the time, but not at pos 0 of source (so each MARKER
    # is followed by a payload token). We'll place them at random positions with
    # the rule: MARKER is always immediately followed by a content token that is
    # the payload to copy.
    is_marker = rng.random(size=(batch, src_len)) < 1 / 3
    # cannot mark last position of source
    is_marker[:, -1] = False
    # place markers in output source
    src = content.copy()
    src = np.where(is_marker, MARKER, src)
    recs[:, 1 : 1 + src_len] = src
    sep_pos = 1 + src_len
    recs[:, sep_pos] = SEP
    # for each sample, collect payload after markers (up to max_out)
    for b in range(batch):
        positions = np.where(is_marker[b])[0]
        payload = content[b, positions + 1][:max_out]
        start = sep_pos + 1
        end = start + len(payload)
        if end > T:
            payload = payload[: T - start]
            end = T
        recs[b, start:end] = payload
        mask[b, start - 1 : end - 1] = True
    return recs, mask


def needle(rng, batch, spec: TaskSpec):
    """Hide a content token in noise, then QUERY and ask for the token."""
    T = spec.seq_len
    recs, mask = _empty(batch, T)
    filler = sample_content(rng, spec, (batch, T))
    recs[:, 1:T] = filler[:, 1:T]
    # needle placed in first half
    needle_pos = rng.integers(5, T // 2, size=batch, dtype=np.int64)
    needle_tok = sample_content(rng, spec, (batch,))
    idx = np.arange(batch)
    recs[idx, needle_pos] = needle_tok
    query_pos = T - 3
    recs[:, query_pos] = QUERY
    recs[idx, query_pos + 1] = needle_tok
    mask[:, query_pos] = True
    return recs, mask


# ---------------------------------------------------------------------------
# Aggregation-style tasks (favour SSMs)
# ---------------------------------------------------------------------------


def counting(rng, batch, spec: TaskSpec):
    """Count occurrences of query token in the body."""
    T = spec.seq_len
    body_len = T // 2
    recs, mask = _empty(batch, T)
    # restrict body to a small alphabet so counts are meaningful
    alpha = min(8, spec.n_content - 1)
    body = rng.integers(CONTENT_LO, CONTENT_LO + alpha, size=(batch, body_len), dtype=np.int32)
    recs[:, 1 : 1 + body_len] = body
    sep_pos = 1 + body_len
    recs[:, sep_pos] = SEP
    q = rng.integers(CONTENT_LO, CONTENT_LO + alpha, size=batch, dtype=np.int32)
    recs[np.arange(batch), sep_pos + 1] = q
    counts = (body == q[:, None]).sum(axis=1)
    counts = np.minimum(counts, spec.n_content - 1)
    recs[np.arange(batch), sep_pos + 2] = (CONTENT_LO + counts).astype(np.int32)
    mask[:, sep_pos + 1] = True
    return recs, mask


def parity(rng, batch, spec: TaskSpec):
    """Parity of the count of a query token in the body."""
    T = spec.seq_len
    body_len = T // 2
    recs, mask = _empty(batch, T)
    alpha = min(8, spec.n_content - 1)
    body = rng.integers(CONTENT_LO, CONTENT_LO + alpha, size=(batch, body_len), dtype=np.int32)
    recs[:, 1 : 1 + body_len] = body
    sep_pos = 1 + body_len
    recs[:, sep_pos] = SEP
    q = rng.integers(CONTENT_LO, CONTENT_LO + alpha, size=batch, dtype=np.int32)
    recs[np.arange(batch), sep_pos + 1] = q
    cnt = (body == q[:, None]).sum(axis=1)
    recs[np.arange(batch), sep_pos + 2] = CONTENT_LO + (cnt % 2).astype(np.int32)
    mask[:, sep_pos + 1] = True
    return recs, mask


def cumulative_sum(rng, batch, spec: TaskSpec, mod: int = 8):
    """Sum a body of small-integer tokens modulo `mod`."""
    T = spec.seq_len
    body_len = T // 2
    recs, mask = _empty(batch, T)
    alpha = 4
    body_vals = rng.integers(0, alpha, size=(batch, body_len), dtype=np.int32)
    body_tok = CONTENT_LO + body_vals
    recs[:, 1 : 1 + body_len] = body_tok
    sep_pos = 1 + body_len
    recs[:, sep_pos] = SEP
    res = (body_vals.sum(axis=1) % mod).astype(np.int32)
    recs[np.arange(batch), sep_pos + 1] = CONTENT_LO + res
    mask[:, sep_pos] = True  # label at sep_pos is the result
    return recs, mask


def state_tracking(rng, batch, spec: TaskSpec, n_states: int = 4, n_inputs: int = 4, seed: int = 12345):
    """Small finite-state automaton; predict final state token."""
    T = spec.seq_len
    body_len = T - 3
    recs, mask = _empty(batch, T)
    # input tokens: [CONTENT_LO, CONTENT_LO+n_inputs)
    # state tokens: [CONTENT_LO+n_inputs, CONTENT_LO+n_inputs+n_states)
    state_rng = np.random.default_rng(seed)
    transitions = state_rng.integers(0, n_states, size=(n_states, n_inputs))
    inputs = rng.integers(0, n_inputs, size=(batch, body_len))
    state = np.zeros(batch, dtype=np.int64)
    for t in range(body_len):
        state = transitions[state, inputs[:, t]]
    recs[:, 1 : 1 + body_len] = CONTENT_LO + inputs
    query_pos = 1 + body_len
    recs[:, query_pos] = QUERY
    recs[np.arange(batch), query_pos + 1] = CONTENT_LO + n_inputs + state.astype(np.int32)
    mask[:, query_pos] = True
    return recs, mask


def mode(rng, batch, spec: TaskSpec):
    """Most-frequent token in the body (ties broken by construction)."""
    T = spec.seq_len
    body_len = T // 2
    recs, mask = _empty(batch, T)
    alpha = 8
    # ensure a unique mode: sample a "boosted" token + uniform noise
    boost = rng.integers(CONTENT_LO, CONTENT_LO + alpha, size=batch, dtype=np.int32)
    body = rng.integers(CONTENT_LO, CONTENT_LO + alpha, size=(batch, body_len), dtype=np.int32)
    # add extra copies of boost token to guarantee it wins
    extra = body_len // 3
    positions = np.stack([rng.permutation(body_len)[:extra] for _ in range(batch)])
    for b in range(batch):
        body[b, positions[b]] = boost[b]
    recs[:, 1 : 1 + body_len] = body
    sep_pos = 1 + body_len
    recs[:, sep_pos] = SEP
    recs[np.arange(batch), sep_pos + 1] = boost
    mask[:, sep_pos] = True
    return recs, mask


def sort_task(rng, batch, spec: TaskSpec):
    """Sort a short sequence of small tokens."""
    T = spec.seq_len
    k = T // 3
    recs, mask = _empty(batch, T)
    alpha = min(16, spec.n_content - 1)
    payload = rng.integers(CONTENT_LO, CONTENT_LO + alpha, size=(batch, k), dtype=np.int32)
    recs[:, 1 : 1 + k] = payload
    recs[:, 1 + k] = SEP
    recs[:, 2 + k : 2 + 2 * k] = np.sort(payload, axis=1)
    mask[:, k + 1 : 2 * k + 1] = True
    return recs, mask
