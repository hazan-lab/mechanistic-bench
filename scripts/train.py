"""Train a single (task, architecture) pair.

Example:
    uv run python scripts/train.py --task induction --arch attn --scale 1m \
        --max_steps 200 --batch_size 64 --seq_len 128
"""

from __future__ import annotations

import argparse

from mechbench.configs import arch_preset, list_archs, list_scales
from mechbench.tasks import list_tasks
from mechbench.training import TrainConfig, train_loop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list_tasks())
    p.add_argument("--arch", required=True, choices=list_archs())
    p.add_argument("--scale", default="1m", choices=list_scales())
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--vocab_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(**vars(args))
    model_cfg = arch_preset(cfg.arch, cfg.scale, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
    train_loop(cfg, model_cfg)


if __name__ == "__main__":
    main()
