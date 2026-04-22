"""Train each task's hard-mode config at 10m-param scale, per architecture.

Reads ``configs/tasks_10m_hard.json``, wraps each task with
``functools.partial`` over the configured hyperparameters, registers the
wrapped callable under a ``<task>_hard10m`` name, and runs the standard
TrainConfig pipeline. Appends per-task results to ``<out_root>/<arch>/status.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

from mechbench.configs import arch_preset, list_archs
from mechbench.tasks.registry import TASK_REGISTRY
from mechbench.training import TrainConfig, train_loop


def register_hard(base_task: str, cfg: dict) -> str:
    base_fn = TASK_REGISTRY[base_task]
    params = cfg.get("task_params", {}) or {}
    wrapped = partial(base_fn, **params) if params else base_fn
    name = f"{base_task}_hard10m"
    TASK_REGISTRY[name] = wrapped
    return name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?", default="configs/tasks_10m_hard.json",
                   help="Path to tasks_10m_hard.json (positional, optional).")
    p.add_argument("out_root", nargs="?",
                   default="/scratch/gpfs/EHAZAN/tharuntk/mechbench/sweep_10m_hard",
                   help="Base output directory (positional, optional).")
    p.add_argument("--archs", nargs="+", default=["attn"], choices=list_archs(),
                   help="Architectures to sweep (one subdir per arch).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip (arch,task) pairs whose run_dir already has a model.pt.")
    return p.parse_args()


def run_arch(arch: str, cfg_path: Path, out_root: Path, skip_existing: bool) -> None:
    arch_root = out_root / arch
    arch_root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(cfg_path.read_text())
    status_path = arch_root / "status.jsonl"
    # append-only when skip_existing (resuming), else truncate.
    mode = "a" if skip_existing else "w"
    status_path.open(mode).close()
    for task, spec_cfg in cfg.items():
        if task.startswith("_"):
            continue
        name = register_hard(task, spec_cfg)
        T = int(spec_cfg["seq_len"])
        V = int(spec_cfg["vocab_size"])
        run_name = f"{task}-{arch}-10m-hard"
        run_dir = arch_root / run_name
        if skip_existing and (run_dir / "model.pt").exists():
            continue
        t0 = time.time()
        with status_path.open("a") as f:
            f.write(json.dumps({"event": "start", "task": task, "arch": arch,
                                "config": spec_cfg, "t": t0}) + "\n")
            f.flush()
        lr = float(spec_cfg.get("lr", 3.0e-4))
        warmup = int(spec_cfg.get("warmup_steps", 200))
        tcfg = TrainConfig(
            task=name, arch=arch, scale="10m",
            seq_len=T, vocab_size=V,
            batch_size=64, eval_batch_size=128, max_steps=4000, warmup_steps=warmup,
            eval_every=400, lr=lr, weight_decay=0.1, grad_clip=1.0, dtype="bf16",
            seed=42, out_dir=str(arch_root), run_name=run_name, log_every=200,
        )
        model_cfg = arch_preset(arch, "10m", seq_len=T, vocab_size=V)
        try:
            history = train_loop(tcfg, model_cfg)
            final = history[-1] if history else {}
        except Exception as e:
            final = {"error": f"{type(e).__name__}: {e}"}
        dt = time.time() - t0
        with status_path.open("a") as f:
            f.write(json.dumps({"event": "end", "task": task, "arch": arch,
                                "t": time.time(), "elapsed_s": dt, "final": final}) + "\n")
            f.flush()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for arch in args.archs:
        run_arch(arch, cfg_path, out_root, args.skip_existing)


if __name__ == "__main__":
    main()
