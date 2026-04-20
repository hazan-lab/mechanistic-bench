"""Sanity checks on task generators: correct shapes and masks."""

from __future__ import annotations

import numpy as np
import pytest

from mechbench.tasks import get_task, list_tasks
from mechbench.tasks.base import TaskSpec


@pytest.mark.parametrize("name", list_tasks())
def test_task_shapes(name):
    spec = TaskSpec(name=name, seq_len=128)
    fn = get_task(name)
    rng = np.random.default_rng(0)
    recs, mask = fn(rng, 4, spec)
    assert recs.shape == (4, 129)
    assert mask.shape == (4, 128)
    assert recs.dtype == np.int32
    assert mask.dtype == bool
    assert (recs >= 0).all() and (recs < spec.vocab_size).all()
    # at least one masked position per sample
    assert (mask.sum(axis=1) > 0).all(), name


@pytest.mark.parametrize("name", list_tasks())
def test_labels_are_valid(name):
    """Where loss_mask is True, the label (next token) must be a content token."""
    spec = TaskSpec(name=name, seq_len=128)
    fn = get_task(name)
    rng = np.random.default_rng(1)
    recs, mask = fn(rng, 8, spec)
    labels = recs[:, 1:][mask]
    from mechbench.tasks.base import CONTENT_LO
    assert (labels >= CONTENT_LO).all(), f"{name}: labels include special tokens"
    assert (labels < spec.vocab_size).all()
