"""Find d_model per architecture that lands closest to a target param count.

Usage:
    uv run python scripts/match_params.py --target 1_000_000 \
        --archs attn mamba stu --n_layers 6 --seq_len 256 --vocab_size 64
"""

from __future__ import annotations

import argparse
from typing import List

import torch

from mechbench.models.model import MechConfig, build_model


def _blocks(arch: str, n_layers: int) -> List[str]:
    return [arch] * n_layers


def count_params(
    arch: str,
    d_model: int,
    n_layers: int,
    n_heads: int,
    seq_len: int,
    vocab_size: int,
    num_eigh: int,
) -> int:
    cfg = MechConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=seq_len,
        block_types=_blocks(arch, n_layers),
        use_flash=False,  # CPU counting — don't need flash-attn
        num_eigh=num_eigh,
    )
    model = build_model(cfg)
    n = sum(p.numel() for p in model.parameters())
    del model
    return n


def best_d_model(
    arch: str,
    target: int,
    n_layers: int,
    n_heads: int,
    seq_len: int,
    vocab_size: int,
    num_eigh: int,
    step: int,
    d_min: int,
    d_max: int,
) -> tuple[int, int]:
    best = None
    for d in range(d_min, d_max + 1, step):
        if d % n_heads != 0:
            continue
        try:
            n = count_params(arch, d, n_layers, n_heads, seq_len, vocab_size, num_eigh)
        except Exception as e:
            print(f"  {arch} d={d} failed: {e}")
            continue
        diff = abs(n - target)
        if best is None or diff < best[0]:
            best = (diff, d, n)
    assert best is not None
    return best[1], best[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=["attn", "mamba", "stu"])
    p.add_argument("--target", type=int, default=1_000_000)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--vocab_size", type=int, default=64)
    p.add_argument("--num_eigh", type=int, default=24)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--d_min", type=int, default=32)
    p.add_argument("--d_max", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    print(
        f"target={args.target:,} n_layers={args.n_layers} n_heads={args.n_heads} "
        f"seq_len={args.seq_len} vocab={args.vocab_size} num_eigh={args.num_eigh}"
    )
    for arch in args.archs:
        d, n = best_d_model(
            arch,
            target=args.target,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            num_eigh=args.num_eigh,
            step=args.step,
            d_min=args.d_min,
            d_max=args.d_max,
        )
        pct = (n - args.target) / args.target * 100
        print(f"  {arch:10s} d_model={d:4d}  params={n:>9,}  ({pct:+.2f}% of target)")


if __name__ == "__main__":
    main()
