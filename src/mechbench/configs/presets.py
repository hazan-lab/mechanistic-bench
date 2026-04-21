"""Architecture presets for the mechanistic suite.

Scales target a rough parameter budget per architecture. We keep n_layers
fixed per scale and vary d_model to land near the budget; exact counts are
resolved via `MechModel.num_parameters` after instantiation.
"""

from __future__ import annotations

from typing import List

from ..models.model import MechConfig


SCALE_LAYERS = {
    "1m": 6,
    "10m": 8,
    "150m": 12,
}

# d_model baseline per scale — adjusted per architecture below.
SCALE_D = {
    "1m": 128,
    "10m": 320,
    "150m": 768,
}


def scale_dmodel(scale: str) -> int:
    return SCALE_D[scale]


def _blocks(pattern: str, n_layers: int) -> List[str]:
    if pattern in {"attn", "mamba", "rnn", "lstm", "mlp"}:
        return [pattern] * n_layers
    if pattern == "alt_attn_mamba":
        return [("attn" if i % 2 == 0 else "mamba") for i in range(n_layers)]
    if pattern == "headwise":
        return ["headwise"] * n_layers
    if pattern == "headwise_alt_attn_mamba":
        # Even layers: headwise (head-level attn+mamba split).
        # Odd layers: alternate pure attn and pure mamba.
        out = []
        for i in range(n_layers):
            if i % 2 == 0:
                out.append("headwise")
            else:
                out.append("attn" if (i // 2) % 2 == 0 else "mamba")
        return out
    raise ValueError(f"Unknown arch pattern {pattern}")


# Per-(scale, arch) d_model offsets, tuned to bring param counts close.
# These are initial guesses; the user can re-tune once measured.
D_MODEL_OFFSETS = {
    "1m": {
        "attn": 0,
        "mamba": -28,
        "rnn": 32,
        "lstm": 0,
        "mlp": 32,
        "alt_attn_mamba": -16,
        "headwise": -16,
    },
    "10m": {
        "attn": 0,
        "mamba": -64,
        "rnn": 64,
        "lstm": 0,
        "mlp": 64,
        "alt_attn_mamba": -32,
        "headwise": -32,
    },
    "150m": {
        "attn": 0,
        "mamba": -96,
        "rnn": 128,
        "lstm": 0,
        "mlp": 128,
        "alt_attn_mamba": -64,
        "headwise": -48,
    },
}


ARCH_NAMES = list(D_MODEL_OFFSETS["1m"].keys())


def list_archs() -> List[str]:
    return list(ARCH_NAMES)


def list_scales() -> List[str]:
    return list(SCALE_LAYERS.keys())


def arch_preset(arch: str, scale: str, seq_len: int = 256, vocab_size: int = 64) -> MechConfig:
    if scale not in SCALE_LAYERS:
        raise KeyError(scale)
    n_layers = SCALE_LAYERS[scale]
    base_d = SCALE_D[scale] + D_MODEL_OFFSETS[scale].get(arch, 0)
    # round d_model to a multiple of n_heads*8 for clean head sizes
    n_heads = {"1m": 4, "10m": 8, "150m": 12}[scale]
    step = max(n_heads * 8, 16)
    d_model = (base_d // step) * step
    d_model = max(d_model, step)
    cfg = MechConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=seq_len,
        block_types=_blocks(arch, n_layers),
        n_attn_heads=max(1, n_heads // 2),
    )
    return cfg
