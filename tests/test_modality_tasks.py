"""Task-specific correctness tests for vision and continuous modalities.

The parametrized tests in ``test_tasks.py`` already check shapes and that
labels are valid content tokens. These tests additionally verify that the
labelled positions actually encode the intended primitive (coordinate
lookup, column parity, 2D patch duplication, fixed-lag echo, denoising,
ordinal nearest-neighbour).
"""

from __future__ import annotations

import numpy as np
import pytest

from mechbench.tasks import get_task
from mechbench.tasks.base import CONTENT_LO, SEP, TaskSpec
from mechbench.tasks.continuous import delayed_echo, nearest_key, piecewise_denoise
from mechbench.tasks.vision import col_parity, grid_retrieval, patch_match


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


def test_grid_retrieval_label_matches_grid_cell():
    spec = TaskSpec(name="grid_retrieval", seq_len=128)
    rng = np.random.default_rng(0)
    recs, mask = grid_retrieval(rng, 32, spec)
    side = int(np.floor(np.sqrt(spec.seq_len - 4)))
    side = max(3, min(side, 10))
    grid_size = side * side
    sep_pos = 1 + grid_size
    for b in range(recs.shape[0]):
        assert recs[b, sep_pos] == SEP
        r = int(recs[b, sep_pos + 1] - CONTENT_LO)
        c = int(recs[b, sep_pos + 2] - CONTENT_LO)
        expected = int(recs[b, 1 + r * side + c])
        # label for input position sep_pos+2 is records[sep_pos+3]
        got = int(recs[b, sep_pos + 3])
        assert got == expected, f"sample {b}: grid[{r},{c}]={expected} but label={got}"
        assert mask[b, sep_pos + 2]


def test_col_parity_label_matches_xor():
    spec = TaskSpec(name="col_parity", seq_len=128)
    rng = np.random.default_rng(1)
    recs, mask = col_parity(rng, 32, spec)
    side = int(np.floor(np.sqrt(spec.seq_len - 4)))
    side = max(3, min(side, 10))
    grid_size = side * side
    sep_pos = 1 + grid_size
    zero = CONTENT_LO
    one = CONTENT_LO + 1
    coord_lo = CONTENT_LO + 2
    for b in range(recs.shape[0]):
        c = int(recs[b, sep_pos + 1] - coord_lo)
        flat = recs[b, 1 : 1 + grid_size]
        col = flat.reshape(side, side)[:, c]
        bits = (col == one).astype(np.int64)
        expected_parity = int(bits.sum() % 2)
        label = int(recs[b, sep_pos + 2])
        assert label in (zero, one)
        assert label - zero == expected_parity
        assert mask[b, sep_pos + 1]


def test_patch_match_both_labels_appear():
    spec = TaskSpec(name="patch_match", seq_len=128)
    rng = np.random.default_rng(2)
    recs, _ = patch_match(rng, 256, spec)
    side = int(np.floor(np.sqrt(spec.seq_len - 3)))
    side = max(3, min(side, 10))
    grid_size = side * side
    sep_pos = 1 + grid_size
    labels = recs[:, sep_pos + 1]
    yes_tok = CONTENT_LO + 1
    no_tok = CONTENT_LO
    n_yes = int((labels == yes_tok).sum())
    n_no = int((labels == no_tok).sum())
    assert n_yes > 30 and n_no > 30, f"label imbalance: yes={n_yes} no={n_no}"


def test_patch_match_duplicate_present_when_labeled_yes():
    spec = TaskSpec(name="patch_match", seq_len=128)
    rng = np.random.default_rng(3)
    recs, _ = patch_match(rng, 128, spec)
    side = int(np.floor(np.sqrt(spec.seq_len - 3)))
    side = max(3, min(side, 10))
    grid_size = side * side
    sep_pos = 1 + grid_size
    yes_tok = CONTENT_LO + 1
    for b in range(recs.shape[0]):
        if recs[b, sep_pos + 1] != yes_tok:
            continue
        grid = recs[b, 1 : 1 + grid_size].reshape(side, side)
        patches = {}
        found_dup = False
        for r in range(side - 1):
            for c in range(side - 1):
                key = tuple(grid[r : r + 2, c : c + 2].reshape(-1).tolist())
                if key in patches:
                    found_dup = True
                    break
                patches[key] = (r, c)
            if found_dup:
                break
        assert found_dup, f"sample {b} labelled YES but no duplicate 2x2 patch found"


# ---------------------------------------------------------------------------
# Continuous
# ---------------------------------------------------------------------------


def test_delayed_echo_output_equals_lagged_input():
    spec = TaskSpec(name="delayed_echo", seq_len=128)
    rng = np.random.default_rng(4)
    lag = 3
    recs, mask = delayed_echo(rng, 16, spec, lag=lag)
    n_pairs = (spec.seq_len - 1) // 2
    for b in range(recs.shape[0]):
        for t in range(lag, n_pairs):
            s_pos = 1 + 2 * t
            assert mask[b, s_pos], f"expected mask True at t={t}"
            sample_prev = int(recs[b, 1 + 2 * (t - lag)])
            echo = int(recs[b, s_pos + 1])
            assert echo == sample_prev
        for t in range(lag):
            s_pos = 1 + 2 * t
            assert not mask[b, s_pos], "pre-warm positions should not contribute to loss"


def test_piecewise_denoise_target_is_cleaner_than_input():
    spec = TaskSpec(name="piecewise_denoise", seq_len=128)
    rng = np.random.default_rng(5)
    recs, mask = piecewise_denoise(rng, 64, spec)
    body_len = (spec.seq_len - 2) // 2
    sep_pos = 1 + body_len
    # mask at position sep_pos refers to record[sep_pos+1]; the target run
    # should be near-constant-piecewise, i.e. fewer transitions than the
    # noisy input on average.
    total_input_switches = 0
    total_target_switches = 0
    for b in range(recs.shape[0]):
        noisy = recs[b, 1 : 1 + body_len]
        out_len = int(mask[b, sep_pos : sep_pos + body_len].sum())
        clean = recs[b, sep_pos + 1 : sep_pos + 1 + out_len]
        total_input_switches += int(np.sum(noisy[1:] != noisy[:-1]))
        total_target_switches += int(np.sum(clean[1:] != clean[:-1]))
    assert total_target_switches < total_input_switches * 0.5, (
        f"denoised target should have fewer transitions "
        f"(input={total_input_switches}, target={total_target_switches})"
    )


def test_nearest_key_label_is_ordinal_nearest():
    spec = TaskSpec(name="nearest_key", seq_len=128)
    rng = np.random.default_rng(6)
    recs, mask = nearest_key(rng, 32, spec)
    n_keys = min(16, spec.n_content // 2)
    val_lo = CONTENT_LO + n_keys
    max_pairs_by_seq = max(2, (spec.seq_len - 4) // 2)
    n_pairs = min(n_keys, max_pairs_by_seq)
    sep_pos = 1 + 2 * n_pairs
    for b in range(recs.shape[0]):
        assert recs[b, sep_pos] == SEP
        kvs = recs[b, 1 : 1 + 2 * n_pairs]
        keys = (kvs[0::2] - CONTENT_LO).astype(np.int64)
        vals = kvs[1::2]
        q = int(recs[b, sep_pos + 1] - CONTENT_LO)
        near_idx = int(np.argmin(np.abs(keys - q)))
        expected_val = int(vals[near_idx])
        got_val = int(recs[b, sep_pos + 2])
        assert got_val == expected_val
        assert got_val >= val_lo
        assert mask[b, sep_pos + 1]


# ---------------------------------------------------------------------------
# Cross-check: the new tasks are reachable through the registry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "grid_retrieval",
        "col_parity",
        "patch_match",
        "delayed_echo",
        "piecewise_denoise",
        "nearest_key",
    ],
)
def test_registry_entry_runs(name):
    spec = TaskSpec(name=name, seq_len=128)
    fn = get_task(name)
    rng = np.random.default_rng(7)
    recs, mask = fn(rng, 4, spec)
    assert recs.shape == (4, 129)
    assert mask.shape == (4, 128)
    assert (mask.sum(axis=1) > 0).all()
