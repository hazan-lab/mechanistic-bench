"""Smoke tests that each architecture preset runs a forward+backward pass."""

from __future__ import annotations

import pytest
import torch

from mechbench.configs import arch_preset, list_archs
from mechbench.models import build_model


@pytest.mark.parametrize("arch", list_archs())
def test_forward_backward(arch):
    cfg = arch_preset(arch, scale="1m", seq_len=64)
    model = build_model(cfg)
    n = model.num_parameters()
    assert n > 0
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, cfg.vocab_size)
    loss = logits.sum()
    loss.backward()
