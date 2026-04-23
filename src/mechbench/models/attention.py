"""Causal self-attention with FlashAttention-2 when available."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import apply_rope, build_rope_cache

try:  # optional: flash-attn-2
    from flash_attn import flash_attn_func  # type: ignore

    HAS_FLASH = True
except Exception:  # pragma: no cover - optional dep
    flash_attn_func = None
    HAS_FLASH = False


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_seq_len: int,
        rope: bool = True,
        use_flash: bool = True,
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope = rope
        self.use_flash = use_flash and HAS_FLASH
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        if rope:
            cos, sin = build_rope_cache(max_seq_len, self.head_dim)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, D)
        if self.rope:
            cos = self.rope_cos[:T].to(q.dtype)
            sin = self.rope_sin[:T].to(q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if self.use_flash and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            # flash-attn expects (B, T, H, D)
            out = flash_attn_func(q, k, v, causal=True)
        else:
            # torch SDPA expects (B, H, T, D)
            q_ = q.transpose(1, 2)
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
            out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, C)
        return self.out(out)


class MultiBranchHeadMixer(nn.Module):
    """Head-wise hybrid: split heads between attention and a Mamba branch.

    For parameter-matching we run attention on ``n_attn_heads`` heads and
    Mamba on the remaining ``n_heads - n_attn_heads`` heads, then concat.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_attn_heads: int,
        max_seq_len: int,
        rope: bool = True,
        use_flash: bool = True,
        mamba_cls=None,
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_variant: str = "mamba",
        mamba2_headdim: int = 64,
        mamba2_chunk_size: int = 256,
    ):
        super().__init__()
        assert 0 < n_attn_heads < n_heads, "n_attn_heads must be between 1 and n_heads-1"
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.n_attn_heads = n_attn_heads
        self.n_mamba_heads = n_heads - n_attn_heads
        self.head_dim = dim // n_heads
        self.attn_dim = self.n_attn_heads * self.head_dim
        self.mamba_dim = self.n_mamba_heads * self.head_dim
        self.attn = CausalSelfAttention(
            self.attn_dim, self.n_attn_heads, max_seq_len, rope=rope, use_flash=use_flash
        )
        if mamba_cls is not None:
            self.mamba = mamba_cls(
                self.mamba_dim, d_state=d_state, d_conv=d_conv, expand=mamba_expand
            )
        elif mamba_variant == "mamba2":
            from .mamba import Mamba2Block
            self.mamba = Mamba2Block(
                self.mamba_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=mamba_expand,
                headdim=mamba2_headdim,
                chunk_size=mamba2_chunk_size,
            )
        else:
            from .mamba import MambaBlock
            self.mamba = MambaBlock(
                self.mamba_dim, d_state=d_state, d_conv=d_conv, expand=mamba_expand
            )
        self.proj_in = nn.Linear(dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        a, m = h.split([self.attn_dim, self.mamba_dim], dim=-1)
        a = self.attn(a)
        m = self.mamba(m)
        h = torch.cat([a, m], dim=-1)
        return self.proj_out(h)
