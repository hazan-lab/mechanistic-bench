"""Train a single (task, architecture) pair.

Two entry forms:

    # flags-only (uses presets.py)
    uv run python scripts/train.py --task induction --arch attn --scale 1m \\
        --max_steps 200 --batch_size 64 --seq_len 128

    # YAML config (model + train fields in one file)
    uv run python scripts/train.py --config configs/scale_1m/transformer.yaml \\
        --task induction

Flag values always win over YAML values when both are supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from mechbench.configs import arch_preset, list_archs, list_scales
from mechbench.models.model import MechConfig
from mechbench.tasks import list_tasks
from mechbench.training import TrainConfig, train_loop


_SENTINEL = object()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="YAML config with model+train fields; flags override")
    p.add_argument("--task", default=_SENTINEL, choices=list_tasks())
    p.add_argument("--arch", default=_SENTINEL, choices=list_archs())
    p.add_argument("--scale", default=_SENTINEL, choices=list_scales())
    p.add_argument("--seq_len", type=int, default=_SENTINEL)
    p.add_argument("--vocab_size", type=int, default=_SENTINEL)
    p.add_argument("--batch_size", type=int, default=_SENTINEL)
    p.add_argument("--eval_batch_size", type=int, default=_SENTINEL)
    p.add_argument("--max_steps", type=int, default=_SENTINEL)
    p.add_argument("--warmup_steps", type=int, default=_SENTINEL)
    p.add_argument("--eval_every", type=int, default=_SENTINEL)
    p.add_argument("--lr", type=float, default=_SENTINEL)
    p.add_argument("--weight_decay", type=float, default=_SENTINEL)
    p.add_argument("--grad_clip", type=float, default=_SENTINEL)
    p.add_argument("--dtype", default=_SENTINEL, choices=["fp32", "bf16", "fp16"])
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=_SENTINEL)
    p.add_argument("--out_dir", default=_SENTINEL)
    p.add_argument("--run_name", default=_SENTINEL)
    p.add_argument("--log_every", type=int, default=_SENTINEL)
    p.add_argument("--wandb", action="store_true", default=False)
    return p.parse_args()


def _load_yaml(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a YAML mapping")
    return data


def _pick(args: argparse.Namespace, key: str, yaml_cfg: dict, default: Any) -> Any:
    v = getattr(args, key, _SENTINEL)
    if v is not _SENTINEL:
        return v
    if key in yaml_cfg:
        return yaml_cfg[key]
    return default


def main():
    args = parse_args()
    yaml_cfg: dict = _load_yaml(args.config) if args.config else {}

    # Build TrainConfig: flags > yaml > dataclass default.
    tc_defaults = {f.name: f.default for f in fields(TrainConfig)}
    tc_kwargs = {}
    for f in fields(TrainConfig):
        val = _pick(args, f.name, yaml_cfg, tc_defaults[f.name])
        tc_kwargs[f.name] = val
    # booleans: argparse store_true leaves default False — only treat True as an override
    if not args.compile and "compile" in yaml_cfg:
        tc_kwargs["compile"] = yaml_cfg["compile"]
    if not args.wandb and "wandb" in yaml_cfg:
        tc_kwargs["wandb"] = yaml_cfg["wandb"]
    cfg = TrainConfig(**tc_kwargs)

    # Build MechConfig: start from preset, then overlay YAML `model:` block.
    model_cfg = arch_preset(cfg.arch, cfg.scale, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
    model_overrides = yaml_cfg.get("model", {}) or {}
    mech_fields = {f.name for f in fields(MechConfig)}
    unknown = set(model_overrides) - mech_fields
    if unknown:
        raise ValueError(f"Unknown model.* keys in {args.config}: {sorted(unknown)}")
    for k, v in model_overrides.items():
        setattr(model_cfg, k, v)
    # Keep vocab/seq_len in sync with TrainConfig.
    model_cfg.vocab_size = cfg.vocab_size
    model_cfg.max_seq_len = cfg.seq_len

    train_loop(cfg, model_cfg)


if __name__ == "__main__":
    main()
