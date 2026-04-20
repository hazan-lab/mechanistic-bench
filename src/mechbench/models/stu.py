"""Spectral Transform Unit (STU) mixer.

Port of flash-stu-2's STU with ``use_approx=True`` and a pure ``torch.fft``
causal convolution (no flash-fft-conv dependency). Kept minimal — just enough
to serve as a drop-in block type in the ``MechModel`` chassis at short
sequence lengths.

References:
- https://github.com/hazan-lab/flash-stu-2 (modules/stu.py, utils/stu_utils.py)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    i = torch.arange(1, seq_len + 1, dtype=torch.float64)
    s = i.unsqueeze(0) + i.unsqueeze(1)  # (L, L), entries start at 2
    if use_hankel_L:
        sgn = (-1.0) ** (s - 2.0) + 1.0
        denom = (s + 3.0) * (s - 1.0) * (s + 1.0)
        return sgn * (8.0 / denom)
    return 2.0 / (s ** 3 - s)


def spectral_filters(seq_len: int, num_eigh: int, use_hankel_L: bool = False) -> torch.Tensor:
    """Top-K eigenvectors of the Hankel matrix, scaled by sigma**0.25.

    Returns a float32 tensor of shape (seq_len, num_eigh).
    """
    Z = _hankel(seq_len, use_hankel_L=use_hankel_L)
    sigma, phi = torch.linalg.eigh(Z)
    phi = phi[:, -num_eigh:] * (sigma[-num_eigh:].clamp_min(0.0) ** 0.25)
    return phi.to(torch.float32)


def _causal_fft_conv(u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Causal 1-D conv along the time axis (dim=1) via real FFT.

    u: (B, L, D), f: (L, D). Pads to next power of two of (2L-1).
    Returns (B, L, D).
    """
    B, L, D = u.shape
    n = 1 << int(math.ceil(math.log2(max(2 * L - 1, 1))))
    Uf = torch.fft.rfft(u, n=n, dim=1)
    Ff = torch.fft.rfft(f, n=n, dim=0)
    Yf = Uf * Ff.unsqueeze(0)
    y = torch.fft.irfft(Yf, n=n, dim=1)
    return y[:, :L, :]


class STUBlock(nn.Module):
    """STU mixer (approximation form).

    For input ``x ∈ (B, L, D_model)``:
      1.   ``x_proj = x @ M_inputs``            → (B, L, D_model)
      2.   ``f = phi @ M_filters``              → (L, D_model)
      3.   ``U_plus  = causal_conv(x_proj, f)``
           ``U_minus = sgn * causal_conv(x_proj * sgn, f)`` with ``sgn_t=(-1)^t``
      4.   out = U_plus + U_minus  (or U_plus if use_hankel_L)

    ``phi`` is a non-learned buffer: top ``num_eigh`` eigenvectors of a
    Hankel matrix, precomputed at construction.
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L

        self.M_inputs = nn.Linear(d_model, d_model, bias=False)
        self.M_filters = nn.Linear(num_eigh, d_model, bias=False)

        phi = spectral_filters(seq_len, num_eigh, use_hankel_L=use_hankel_L)  # (L, K)
        self.register_buffer("phi", phi, persistent=False)
        sgn = torch.ones(seq_len)
        sgn[1::2] = -1.0
        self.register_buffer("sgn", sgn.view(1, seq_len, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        assert L <= self.seq_len, f"STU seq_len={self.seq_len} but got L={L}"
        phi = self.phi[:L]              # (L, K)
        sgn = self.sgn[:, :L, :]        # (1, L, 1)

        x_proj = self.M_inputs(x)                         # (B, L, D)
        f = self.M_filters(phi).to(x_proj.dtype)          # (L, D)

        # Compute in fp32 for FFT stability, then cast back.
        xp = x_proj.float()
        ff = f.float()
        U_plus = _causal_fft_conv(xp, ff)
        if self.use_hankel_L:
            out = U_plus
        else:
            U_minus = sgn * _causal_fft_conv(xp * sgn, ff)
            out = U_plus + U_minus
        return out.to(x_proj.dtype)
