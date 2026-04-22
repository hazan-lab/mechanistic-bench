"""Mamba mixer — delegates to `mamba_ssm` if installed, otherwise falls back to
a pure-PyTorch *minimal* selective-SSM implementation suitable for short
mechanistic-bench sequences (~256 tokens).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _MambaSSM  # type: ignore

    HAS_MAMBA_SSM = True
except Exception:  # pragma: no cover
    _MambaSSM = None
    HAS_MAMBA_SSM = False

try:
    from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2SSM  # type: ignore

    HAS_MAMBA2_SSM = True
except Exception:  # pragma: no cover
    _Mamba2SSM = None
    HAS_MAMBA2_SSM = False


class MinimalMamba(nn.Module):
    """A pure-PyTorch selective SSM with a reference (slow) sequential scan.

    Enough for correctness at short sequences; replace with `mamba_ssm.Mamba`
    for real training once it is available.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # A: log-space, initialised to -exp; D: skip connection
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_and_res = self.in_proj(x)
        x_inner, res = x_and_res.chunk(2, dim=-1)  # (B, T, d_inner)
        # causal conv
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :T]
        x_inner = x_inner.transpose(1, 2)
        x_inner = F.silu(x_inner)
        # compute Δ, B_, C_
        x_proj = self.x_proj(x_inner)  # (B, T, d_state*2 + d_inner)
        delta = x_proj[..., : self.d_inner]
        Bm = x_proj[..., self.d_inner : self.d_inner + self.d_state]
        Cm = x_proj[..., self.d_inner + self.d_state :]
        delta = F.softplus(self.dt_proj(delta))  # (B, T, d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # selective scan (sequential, slow but correct). Cast to float32 for stability.
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, T, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * Bm.unsqueeze(-2) * x_inner.unsqueeze(-1)  # (B,T,d_inner,d_state)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=torch.float32)
        outs = []
        for t in range(T):
            h = deltaA[:, t].float() * h + deltaB_u[:, t].float()
            y = (h * Cm[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            outs.append(y)
        y = torch.stack(outs, dim=1)  # (B, T, d_inner)
        y = y.to(x.dtype) + x_inner * self.D
        y = y * F.silu(res)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Mamba mixer that prefers mamba_ssm, falls back to MinimalMamba."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if HAS_MAMBA_SSM:
            self.inner = _MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.inner = MinimalMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


class Mamba2Block(nn.Module):
    """Mamba-2 (SSD) mixer. Requires mamba_ssm >= 2.0 — no fallback.

    Note: Mamba-2 requires d_inner (= expand * d_model) to be divisible by
    headdim, and chunk_size should not exceed the sequence length. The
    shipped scale_1m/mamba2.yaml uses d_model=96, d_state=256, headdim=64,
    expand=2, chunk_size=256 to hit ~1M params.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
    ):
        super().__init__()
        if not HAS_MAMBA2_SSM:
            raise ImportError(
                "Mamba2Block requires mamba_ssm>=2.0 with the Mamba2 module; "
                "install with `uv pip install mamba-ssm`."
            )
        d_inner = expand * d_model
        if d_inner % headdim != 0:
            raise ValueError(
                f"d_inner={d_inner} (expand*d_model={expand}*{d_model}) must be "
                f"divisible by headdim={headdim}"
            )
        self.inner = _Mamba2SSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)
