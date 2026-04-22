"""Unified sequence-model chassis.

The same ``MechModel`` is used for all architectures; it stacks a list of
mixer blocks specified by the ``MechConfig.block_types`` list. Supported block
types:

- ``"attn"``      causal self-attention (RoPE, flash-attn when available)
- ``"mamba"``     Mamba SSM mixer
- ``"rnn"``       Elman RNN
- ``"lstm"``      LSTM
- ``"mlp"``       token-wise MLP (no cross-token mixing)
- ``"headwise"``  head-wise attention+mamba hybrid (split heads)

For layer-wise hybrids, simply alternate block types in ``block_types``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention, MultiBranchHeadMixer
from .common import RMSNorm, SwiGLU
from .mamba import Mamba2Block, MambaBlock
from .recurrent import ElmanRNN, LSTMMixer, MLPMixer
from .stu import STUBlock


@dataclass
class MechConfig:
    vocab_size: int = 64
    d_model: int = 128
    n_heads: int = 4
    max_seq_len: int = 256
    mlp_hidden_mult: float = 2.0
    block_types: List[str] = field(default_factory=lambda: ["attn"] * 6)
    # mamba hyperparams
    d_state: int = 16
    d_conv: int = 4
    mamba_expand: int = 2
    # mamba-2 hyperparams (Mamba2 block only)
    mamba2_headdim: int = 64
    mamba2_chunk_size: int = 256
    # attention opts
    rope: bool = True
    use_flash: bool = True
    # headwise hybrid
    n_attn_heads: int = 2   # used only when block_type == "headwise"
    # stu
    num_eigh: int = 24
    use_hankel_L: bool = False
    # regularisation
    dropout: float = 0.0
    tie_embeddings: bool = True


def _build_mixer(block_type: str, cfg: MechConfig) -> nn.Module:
    if block_type == "attn":
        return CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.max_seq_len, rope=cfg.rope, use_flash=cfg.use_flash
        )
    if block_type == "mamba":
        return MambaBlock(cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.mamba_expand)
    if block_type == "mamba2":
        return Mamba2Block(
            cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.mamba_expand,
            headdim=cfg.mamba2_headdim,
            chunk_size=cfg.mamba2_chunk_size,
        )
    if block_type == "rnn":
        return ElmanRNN(cfg.d_model)
    if block_type == "lstm":
        return LSTMMixer(cfg.d_model)
    if block_type == "mlp":
        return MLPMixer(cfg.d_model, hidden_mult=cfg.mlp_hidden_mult)
    if block_type == "stu":
        return STUBlock(
            cfg.d_model,
            seq_len=cfg.max_seq_len,
            num_eigh=cfg.num_eigh,
            use_hankel_L=cfg.use_hankel_L,
        )
    if block_type == "headwise":
        return MultiBranchHeadMixer(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_attn_heads,
            cfg.max_seq_len,
            rope=cfg.rope,
            use_flash=cfg.use_flash,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            mamba_expand=cfg.mamba_expand,
        )
    raise ValueError(f"Unknown block_type '{block_type}'")


class Block(nn.Module):
    def __init__(self, cfg: MechConfig, block_type: str):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = _build_mixer(block_type, cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, hidden=int(cfg.mlp_hidden_mult * cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mixer(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class MechModel(nn.Module):
    def __init__(self, cfg: MechConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, bt) for bt in cfg.block_types])
        self.norm = RMSNorm(cfg.d_model)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        if self.lm_head is None:
            logits = F.linear(h, self.embed.weight)
        else:
            logits = self.lm_head(h)
        return logits

    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))


def build_model(cfg: MechConfig) -> MechModel:
    return MechModel(cfg)
