"""Train each task's hard-mode config at 10m-param attn scale.

Reads ``configs/tasks_10m_hard.json``, wraps each task with
``functools.partial`` over the configured hyperparameters, registers the
wrapped callable under a ``<task>_hard10m`` name, and runs the standard
TrainConfig pipeline. Appends per-task results to ``status.jsonl``.
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

from mechbench.configs import arch_preset
from mechbench.tasks.registry import TASK_REGISTRY
from mechbench.training import TrainConfig, train_loop


def register_hard(base_task: str, cfg: dict) -> str:
    base_fn = TASK_REGISTRY[base_task]
    params = cfg.get("task_params", {}) or {}
    wrapped = partial(base_fn, **params) if params else base_fn
    name = f"{base_task}_hard10m"
    TASK_REGISTRY[name] = wrapped
    return name


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/tasks_10m_hard.json")
    out_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/sweep_10m_hard")
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(cfg_path.read_text())
    status_path = out_root / "status.jsonl"
    status_path.write_text("")
    for task, spec_cfg in cfg.items():
        if task.startswith("_"):
            continue
        name = register_hard(task, spec_cfg)
        T = int(spec_cfg["seq_len"])
        V = int(spec_cfg["vocab_size"])
        t0 = time.time()
        with status_path.open("a") as f:
            f.write(json.dumps({"event": "start", "task": task, "config": spec_cfg, "t": t0}) + "\n")
            f.flush()
        lr = float(spec_cfg.get("lr", 3.0e-4))
        warmup = int(spec_cfg.get("warmup_steps", 200))
        tcfg = TrainConfig(
            task=name, arch="attn", scale="10m",
            seq_len=T, vocab_size=V,
            batch_size=64, eval_batch_size=128, max_steps=4000, warmup_steps=warmup,
            eval_every=400, lr=lr, weight_decay=0.1, grad_clip=1.0, dtype="bf16",
            seed=42, out_dir=str(out_root), run_name=f"{task}-attn-10m-hard", log_every=200,
        )
        model_cfg = arch_preset("attn", "10m", seq_len=T, vocab_size=V)
        try:
            history = train_loop(tcfg, model_cfg)
            final = history[-1] if history else {}
        except Exception as e:
            final = {"error": str(e)}
        dt = time.time() - t0
        with status_path.open("a") as f:
            f.write(json.dumps({"event": "end", "task": task, "t": time.time(),
                                 "elapsed_s": dt, "final": final}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
