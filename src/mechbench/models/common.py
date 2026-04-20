"""Shared components: RMSNorm, SwiGLU MLP, RoPE, embeddings."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or int(8 / 3 * dim)
        hidden = (hidden + 7) // 8 * 8
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=torch.float32):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    angles = torch.outer(t, freqs)  # (T, half)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
    """Rotate last dim in halves. ``seq_dim`` is the axis indexing time."""
    assert x.dim() == 4
    T = x.shape[seq_dim]
    assert cos.shape[0] >= T, (cos.shape, T)
    cos = cos[:T]
    sin = sin[:T]
    shape = [1] * x.dim()
    shape[seq_dim] = T
    shape[-1] = x.shape[-1] // 2
    cos_ = cos.reshape(shape)
    sin_ = sin.reshape(shape)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    out1 = x1 * cos_ - x2 * sin_
    out2 = x1 * sin_ + x2 * cos_
    return torch.cat([out1, out2], dim=-1)
