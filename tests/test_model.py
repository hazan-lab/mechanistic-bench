"""Smoke tests that each architecture preset runs a forward+backward pass."""

from __future__ import annotations

import pytest
import torch

from mechbench.configs import arch_preset, list_archs
from mechbench.models import build_model

_CUDA_ONLY_ARCHS = {"mamba", "mamba2", "alt_attn_mamba", "headwise"}


@pytest.mark.parametrize("arch", list_archs())
def test_forward_backward(arch):
    if arch in _CUDA_ONLY_ARCHS and not torch.cuda.is_available():
        pytest.skip(f"{arch} requires a CUDA device (mamba_ssm / causal_conv1d kernels)")
    device = torch.device("cuda" if arch in _CUDA_ONLY_ARCHS else "cpu")
    cfg = arch_preset(arch, scale="1m", seq_len=64)
    model = build_model(cfg).to(device)
    n = model.num_parameters()
    assert n > 0
    x = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    logits = model(x)
    assert logits.shape == (2, 32, cfg.vocab_size)
    loss = logits.sum()
    loss.backward()
