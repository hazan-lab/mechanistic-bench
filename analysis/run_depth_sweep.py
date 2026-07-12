"""Compositional-depth sweep for Q2.

Trains 5 architectures × 5 hop-depths (k=1..5) on deep_hop at the 1m
scale. Each run is short (max_steps=2000) since 1m models plateau by
~step 1500 on hop tasks.

Outputs:
    workshop_analysis/depth_sweep.csv  (rows: arch, k, tok_acc, eval_loss, n_params)
"""
from __future__ import annotations
import json
import sys
import time
import yaml
import gc
from pathlib import Path
from functools import partial

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechbench.training import TrainConfig, train_loop
from mechbench.models.model import MechConfig
from mechbench.tasks.registry import get_task

OUT = Path(__file__).resolve().parent / "depth_sweep.csv"
RUN_ROOT = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/depth_sweep")
RUN_ROOT.mkdir(parents=True, exist_ok=True)

ARCHS = ["transformer", "mamba2", "stu", "alt_attn_mamba", "headwise"]
DEPTHS = [1, 2, 3, 4, 5]
MAX_STEPS = 2000


def load_cfgs(arch: str):
    yml = REPO / "configs" / "scale_1m" / "models" / f"{arch}.yaml"
    if not yml.exists():
        # try main repo path
        yml = Path("/home/tt6444/mechanistic-bench") / "configs" / "scale_1m" / "models" / f"{arch}.yaml"
    d = yaml.safe_load(yml.read_text())
    model = d["model"]
    train_kwargs = {k: v for k, v in d.items() if k not in ("model", "arch", "scale")}
    train_kwargs["arch"] = d.get("arch", arch)
    train_kwargs["scale"] = d.get("scale", "1m")
    return train_kwargs, model


def build_train_cfg(arch_yaml: dict, task: str, k: int, max_steps: int) -> TrainConfig:
    cfg = TrainConfig(
        task=task,
        arch=arch_yaml["arch"],
        scale="1m",
        seq_len=256,
        vocab_size=64,
        batch_size=arch_yaml.get("batch_size", 128),
        eval_batch_size=arch_yaml.get("eval_batch_size", 256),
        eval_every=arch_yaml.get("eval_every", 500),
        max_steps=max_steps,
        warmup_steps=arch_yaml.get("warmup_steps", 200),
        lr=arch_yaml.get("lr", 3e-4),
        weight_decay=arch_yaml.get("weight_decay", 0.1),
        grad_clip=arch_yaml.get("grad_clip", 1.0),
        dtype=arch_yaml.get("dtype", "bf16"),
        seed=arch_yaml.get("seed", 42),
        out_dir=str(RUN_ROOT),
        run_name=f"deep_hop_k{k}-{arch_yaml['arch']}-1m",
        log_every=arch_yaml.get("log_every", 200),
    )
    return cfg


def main():
    rows = []
    if OUT.exists():
        for line in OUT.read_text().splitlines()[1:]:
            parts = line.split(",")
            rows.append({"arch": parts[0], "k": int(parts[1]), "tok_acc": float(parts[2]),
                         "eval_loss": float(parts[3])})
    done = {(r["arch"], r["k"]) for r in rows}
    print(f"already done: {len(done)} runs")

    for arch in ARCHS:
        train_kwargs, model_kwargs = load_cfgs(arch)
        # Need n_content for task generation; deep_hop with n_nodes=10 needs k+3 reserve and 1+2*10+3+k positions
        # max k=5 needs T=1+20+3+5=29 -> seq_len=256 OK
        for k in DEPTHS:
            arch_name = train_kwargs["arch"]
            if (arch_name, k) in done:
                print(f"[skip] {arch_name} k={k}")
                continue
            t0 = time.time()
            print(f"\n=== {arch_name} k={k} ===")
            base_fn = get_task("deep_hop")
            task_fn = partial(base_fn, n_nodes=10, k=k)

            cfg = build_train_cfg(train_kwargs, "deep_hop", k, MAX_STEPS)
            mech_cfg = MechConfig(
                **model_kwargs, max_seq_len=cfg.seq_len, vocab_size=cfg.vocab_size,
            )
            try:
                train_loop(cfg, mech_cfg, task_fn=task_fn)
            except Exception as e:
                print(f"!! failed: {e}")
                continue
            # Read history
            hist_path = RUN_ROOT / cfg.run_name / "history.json"
            try:
                hist = json.load(open(hist_path))["history"]
                last = hist[-1]
                rows.append({
                    "arch": arch_name, "k": k,
                    "tok_acc": last.get("tok_acc"),
                    "eval_loss": last.get("eval_loss"),
                })
            except Exception as e:
                print(f"!! history read failed: {e}")
            elapsed = time.time() - t0
            print(f"   done in {elapsed:.0f}s -> tok_acc={rows[-1]['tok_acc']:.3f}")
            torch.cuda.empty_cache()
            gc.collect()

            # Write incremental CSV
            with OUT.open("w") as f:
                f.write("arch,k,tok_acc,eval_loss\n")
                for r in rows:
                    f.write(f"{r['arch']},{r['k']},{r['tok_acc']:.6f},{r['eval_loss']:.6f}\n")

    print(f"\nfinal -> {OUT}")
    for r in rows:
        print(f"  {r}")


if __name__ == "__main__":
    main()
