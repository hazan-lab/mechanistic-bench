"""Find the d_model that makes each arch closest to 150M params at LM scale.

Instantiates a CPU-only MechModel for each (arch, d_model) candidate and
counts parameters. Prints a table and the best d_model per arch.

Usage:
    uv run python scripts/tune_150m.py
"""

from __future__ import annotations

import argparse

import torch

from mechbench.configs.presets import _blocks
from mechbench.models.model import MechConfig, MechModel


TARGET = 150_000_000
N_LAYERS = 12
N_HEADS = 12
MLP_HIDDEN_MULT = 4.0  # standard transformer
VOCAB = 50304
SEQ_LEN = 2048
ARCHS = ("attn", "mamba", "alt_attn_mamba")
# Finer step: multiple of n_heads*2 so head_dim stays even.
D_STEP = N_HEADS * 4  # 48  -> head_dim in {16, 20, 24, ...}


def _round(d: int, step: int) -> int:
    return max(step, (d // step) * step)


def count_params(arch: str, d_model: int, mlp_mult: float = MLP_HIDDEN_MULT) -> tuple[int, int]:
    d_model = _round(d_model, D_STEP)
    cfg = MechConfig(
        vocab_size=VOCAB,
        d_model=d_model,
        n_heads=N_HEADS,
        max_seq_len=SEQ_LEN,
        block_types=_blocks(arch, N_LAYERS),
        n_attn_heads=max(1, N_HEADS // 2),
        mlp_hidden_mult=mlp_mult,
        tie_embeddings=True,
        use_flash=False,
    )
    with torch.device("meta"):
        model = MechModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    return d_model, n


def search(arch: str, lo: int = 256, hi: int = 1600) -> tuple[int, float, int]:
    """Return (d_model, mlp_hidden_mult, params) closest to TARGET."""
    best = None  # (delta, d, mult, n)
    mults = [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
    d = lo
    while d <= hi:
        for mult in mults:
            d_used, n = count_params(arch, d, mlp_mult=mult)
            delta = abs(n - TARGET)
            if best is None or delta < best[0]:
                best = (delta, d_used, mult, n)
        d += D_STEP
    return best[1], best[2], best[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Target: {TARGET/1e6:.1f}M params | layers={N_LAYERS} heads={N_HEADS} vocab={VOCAB} seq_len={SEQ_LEN}")
    print()
    print(f"{'arch':>30s}  {'d_model':>8s}  {'mlp_mult':>8s}  {'params':>12s}  {'delta':>10s}")
    print("-" * 80)
    rows = []
    for arch in ARCHS:
        d, mult, n = search(arch)
        pct = 100.0 * (n - TARGET) / TARGET
        rows.append((arch, d, mult, n, pct))
        print(f"{arch:>30s}  {d:>8d}  {mult:>8.2f}  {n/1e6:>10.2f}M  {pct:>+8.2f}%")

    print()
    print("YAML block_types per arch (for reference):")
    for arch, _, _, _, _ in rows:
        blocks = _blocks(arch, N_LAYERS)
        print(f"  {arch}: {blocks}")


if __name__ == "__main__":
    main()
