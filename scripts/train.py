"""Train a single (model, task) pair at a given scale.

Usage:
    uv run python scripts/train.py --scale 1m --model transformer --task induction

All run settings come from the two YAMLs:
    configs/scale_<scale>/models/<model>.yaml   (arch + training defaults + model block)
    configs/scale_<scale>/tasks.yaml            (per-task seq_len, vocab_size, task_params,
                                                 plus optional training overrides)

Merge order (lowest → highest precedence):
    model YAML  <  tasks.defaults  <  tasks.<name>  <  CLI flags

The fully resolved config is dumped to ``<out_dir>/<run_name>/config.yaml`` at
the start of every run so each artifact is self-describing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from mechbench.configs import list_scales
from mechbench.models.model import MechConfig
from mechbench.tasks import list_tasks
from mechbench.tasks.registry import get_task
from mechbench.training import TrainConfig, train_loop


_SENTINEL = object()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", required=True, choices=list_scales())
    p.add_argument("--model", required=True,
                   help="Model YAML basename under configs/scale_<scale>/models/ (no .yaml)")
    p.add_argument("--task", required=True, choices=list_tasks())
    # Optional CLI overrides — usually for smoke tests. Values land in the
    # dumped config.yaml so the run artifact remains self-describing.
    p.add_argument("--max_steps", type=int, default=_SENTINEL)
    p.add_argument("--batch_size", type=int, default=_SENTINEL)
    p.add_argument("--lr", type=float, default=_SENTINEL)
    p.add_argument("--seed", type=int, default=_SENTINEL)
    p.add_argument("--dtype", default=_SENTINEL, choices=["fp32", "bf16", "fp16"])
    p.add_argument("--out_dir", default=_SENTINEL)
    p.add_argument("--run_name", default=_SENTINEL)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--wandb", action="store_true", default=False)
    return p.parse_args()


def _resolve_configs(args: argparse.Namespace) -> tuple[TrainConfig, MechConfig, dict, str]:
    root = _repo_root()
    model_yaml_path = root / "configs" / f"scale_{args.scale}" / "models" / f"{args.model}.yaml"
    tasks_yaml_path = root / "configs" / f"scale_{args.scale}" / "tasks.yaml"
    if not model_yaml_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_yaml_path}")
    if not tasks_yaml_path.exists():
        raise FileNotFoundError(f"Tasks YAML not found: {tasks_yaml_path}")

    model_cfg_raw = _load_yaml(model_yaml_path)
    tasks_cfg_raw = _load_yaml(tasks_yaml_path)

    # Consistency check: model YAML's declared scale must match the directory.
    if model_cfg_raw.get("scale") != args.scale:
        raise ValueError(
            f"{model_yaml_path} declares scale={model_cfg_raw.get('scale')!r} "
            f"but lives under scale_{args.scale}/"
        )

    task_defaults = tasks_cfg_raw.get("defaults", {}) or {}
    tasks_map = tasks_cfg_raw.get("tasks", {}) or {}
    if args.task not in tasks_map:
        raise KeyError(
            f"Task {args.task!r} not listed in {tasks_yaml_path}. "
            f"Available: {sorted(tasks_map.keys())}"
        )
    task_entry = tasks_map[args.task] or {}

    # Start from TrainConfig defaults, overlay model YAML, then task defaults,
    # then per-task entry, then CLI flags.
    tc_defaults = {f.name: f.default for f in fields(TrainConfig)}
    tc_kwargs: dict = dict(tc_defaults)

    # Layer 1: model YAML top-level training fields (skip "model:" block).
    model_block = model_cfg_raw.get("model", {}) or {}
    for k, v in model_cfg_raw.items():
        if k == "model":
            continue
        if k in tc_kwargs:
            tc_kwargs[k] = v

    # Layer 2: tasks.defaults.
    for k, v in task_defaults.items():
        if k == "task_params":
            continue
        if k not in tc_kwargs:
            raise ValueError(f"Unknown key {k!r} in tasks.defaults (not a TrainConfig field)")
        tc_kwargs[k] = v

    # Layer 3: per-task entry.
    for k, v in task_entry.items():
        if k == "task_params":
            continue
        if k not in tc_kwargs:
            raise ValueError(
                f"Unknown key {k!r} in tasks.{args.task} (not a TrainConfig field)"
            )
        tc_kwargs[k] = v

    tc_kwargs["task"] = args.task

    # Layer 4: CLI overrides (only those explicitly supplied).
    for cli_key in ("max_steps", "batch_size", "lr", "seed", "dtype",
                    "out_dir", "run_name"):
        v = getattr(args, cli_key)
        if v is not _SENTINEL:
            tc_kwargs[cli_key] = v
    if args.compile:
        tc_kwargs["compile"] = True
    if args.wandb:
        tc_kwargs["wandb"] = True

    cfg = TrainConfig(**tc_kwargs)

    # Build MechConfig from the model block; seq_len/vocab_size flow from the
    # task, so max_seq_len is always consistent with the data.
    mech_fields = {f.name for f in fields(MechConfig)}
    unknown = set(model_block) - mech_fields
    if unknown:
        raise ValueError(f"Unknown model.* keys in {model_yaml_path}: {sorted(unknown)}")
    mech_cfg = MechConfig()
    for k, v in model_block.items():
        setattr(mech_cfg, k, v)
    mech_cfg.vocab_size = cfg.vocab_size
    mech_cfg.max_seq_len = cfg.seq_len

    # Resolve task_params with proper precedence (defaults < per-task).
    task_params = dict(task_defaults.get("task_params") or {})
    task_params.update(task_entry.get("task_params") or {})

    # Validate task_params against the generator signature. Generator args
    # are (rng, batch, spec, **kwargs); anything else must be a real kwarg.
    import inspect
    base_fn = get_task(args.task)
    try:
        sig = inspect.signature(base_fn)
    except (TypeError, ValueError):
        sig = None  # builtins / partials without usable signatures
    if sig is not None:
        accepted = {name for name in sig.parameters if name not in {"rng", "batch", "spec"}}
        unknown_params = set(task_params) - accepted
        if unknown_params:
            raise ValueError(
                f"task_params for {args.task} include unknown keyword(s): {sorted(unknown_params)}"
            )

    return cfg, mech_cfg, task_params, str(model_yaml_path.relative_to(root))


def _dump_resolved_config(out_dir: Path, cfg: TrainConfig, mech_cfg: MechConfig,
                          task_params: dict, model_yaml_rel: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "source": {
            "model_yaml": model_yaml_rel,
            "tasks_yaml": f"configs/scale_{cfg.scale}/tasks.yaml",
        },
        "train": asdict(cfg),
        "model": asdict(mech_cfg),
        "task_params": task_params,
    }
    (out_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))


def main() -> None:
    args = parse_args()
    cfg, mech_cfg, task_params, model_yaml_rel = _resolve_configs(args)

    run_name = cfg.run_name or f"{cfg.task}-{cfg.arch}-{cfg.scale}"
    out_dir = Path(cfg.out_dir) / run_name
    _dump_resolved_config(out_dir, cfg, mech_cfg, task_params, model_yaml_rel)

    base_fn = get_task(args.task)
    task_fn = partial(base_fn, **task_params) if task_params else base_fn

    train_loop(cfg, mech_cfg, task_fn=task_fn)


if __name__ == "__main__":
    main()
