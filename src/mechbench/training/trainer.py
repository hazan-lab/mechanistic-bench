"""Training loop for a single (task, architecture) pair."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..models.model import MechModel, MechConfig, build_model
from ..tasks.base import TaskSpec, records_to_batch
from ..tasks.registry import get_task
from .schedule import cosine_with_warmup


@dataclass
class TrainConfig:
    task: str = "induction"
    arch: str = "attn"
    scale: str = "1m"
    seq_len: int = 256
    vocab_size: int = 64
    batch_size: int = 128
    eval_batch_size: int = 256
    eval_every: int = 500
    max_steps: int = 4000
    warmup_steps: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dtype: str = "bf16"   # "fp32" | "bf16" | "fp16"
    compile: bool = False
    seed: int = 42
    out_dir: str = "/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs"
    run_name: Optional[str] = None
    log_every: int = 50
    wandb: bool = False
    project: str = "mechbench"


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _make_batch(task_fn, rng: np.random.Generator, cfg: TrainConfig, spec: TaskSpec, device):
    recs, mask = task_fn(rng, cfg.batch_size, spec)
    return records_to_batch(recs, mask, device=device)


@torch.no_grad()
def evaluate(model: MechModel, task_fn, rng, cfg: TrainConfig, spec: TaskSpec, device, n_batches: int = 4):
    model.eval()
    correct_tok = 0
    total_tok = 0
    correct_seq = 0
    total_seq = 0
    loss_sum = 0.0
    loss_tok = 0
    for _ in range(n_batches):
        recs, mask = task_fn(rng, cfg.eval_batch_size, spec)
        batch = records_to_batch(recs, mask, device=device)
        with torch.autocast(device_type="cuda", dtype=_dtype(cfg.dtype), enabled=(cfg.dtype != "fp32")):
            logits = model(batch.inputs)
        preds = logits.argmax(dim=-1)
        valid = batch.loss_mask
        labels = torch.from_numpy(recs.astype(np.int64)).to(device)[:, 1:]
        eq = (preds == labels) & valid
        correct_tok += eq.sum().item()
        total_tok += valid.sum().item()
        # sequence accuracy: all masked positions correct within a row
        per_row_correct = eq.sum(dim=1)
        per_row_total = valid.sum(dim=1)
        seq_ok = (per_row_correct == per_row_total) & (per_row_total > 0)
        correct_seq += seq_ok.sum().item()
        total_seq += (per_row_total > 0).sum().item()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.targets.reshape(-1), ignore_index=-100, reduction="sum")
        loss_sum += loss.item()
        loss_tok += valid.sum().item()
    model.train()
    tok_acc = correct_tok / max(1, total_tok)
    seq_acc = correct_seq / max(1, total_seq)
    eval_loss = loss_sum / max(1, loss_tok)
    eval_ppl = math.exp(min(eval_loss, 50.0))
    return {"tok_acc": tok_acc, "seq_acc": seq_acc, "eval_loss": eval_loss, "eval_ppl": eval_ppl}


def train_loop(cfg: TrainConfig, model_cfg: MechConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_name = cfg.run_name or f"{cfg.task}-{cfg.arch}-{cfg.scale}"
    out_dir = Path(cfg.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_cfg).to(device)
    if cfg.dtype == "bf16":
        # keep params fp32, autocast handles the cast
        pass
    n_params = model.num_parameters()
    print(f"[{run_name}] arch={cfg.arch} d_model={model_cfg.d_model} layers={len(model_cfg.block_types)} params={n_params/1e6:.2f}M")

    if cfg.compile:
        model = torch.compile(model)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )
    scheduler = cosine_with_warmup(optim, cfg.warmup_steps, cfg.max_steps)

    task_fn = get_task(cfg.task)
    spec = TaskSpec(name=cfg.task, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
    train_rng = np.random.default_rng(cfg.seed)
    eval_rng = np.random.default_rng(cfg.seed + 1)

    use_wandb = False
    if cfg.wandb:
        try:
            import wandb
            wandb.init(project=cfg.project, name=run_name, config={**asdict(cfg), "model": asdict(model_cfg), "n_params": n_params})
            use_wandb = True
        except Exception as e:  # pragma: no cover
            print(f"wandb unavailable: {e}")

    history: list[dict] = []
    train_log: list[dict] = []
    step = 0
    t0 = time.time()
    # Accumulate on GPU; only sync at log_every to avoid per-step pipeline stalls.
    loss_sum_gpu = torch.zeros((), device=device, dtype=torch.float32)
    tok_sum_gpu = torch.zeros((), device=device, dtype=torch.float32)
    while step < cfg.max_steps:
        batch = _make_batch(task_fn, train_rng, cfg, spec, device)
        with torch.autocast(device_type="cuda", dtype=_dtype(cfg.dtype), enabled=(cfg.dtype != "fp32")):
            logits = model(batch.inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.targets.reshape(-1),
                ignore_index=-100,
            )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        scheduler.step()

        # GPU-side accumulation — no sync on hot path.
        tok_count = batch.loss_mask.sum().to(torch.float32)
        loss_sum_gpu += loss.detach().to(torch.float32) * tok_count
        tok_sum_gpu += tok_count
        step += 1

        if step % cfg.log_every == 0:
            # Single sync for the log interval.
            ls = loss_sum_gpu.item()
            ts = tok_sum_gpu.item()
            mean_loss = ls / max(1.0, ts)
            mean_ppl = math.exp(min(mean_loss, 50.0))
            tps = cfg.batch_size * cfg.seq_len * cfg.log_every / max(1e-9, time.time() - t0)
            cur_lr = scheduler.get_last_lr()[0]
            print(f"step {step}/{cfg.max_steps}  loss {mean_loss:.4f}  ppl {mean_ppl:.2f}  lr {cur_lr:.2e}  tok/s {tps:.0f}")
            train_log.append({"step": step, "train_loss": mean_loss, "train_ppl": mean_ppl, "lr": cur_lr, "tok_per_s": tps})
            (out_dir / "train_log.json").write_text(json.dumps({"train_log": train_log, "n_params": n_params}, indent=2))
            if use_wandb:
                import wandb
                wandb.log({"train/loss": mean_loss, "train/ppl": mean_ppl, "train/lr": cur_lr, "step": step})
            loss_sum_gpu.zero_()
            tok_sum_gpu.zero_()
            t0 = time.time()

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            metrics = evaluate(model, task_fn, eval_rng, cfg, spec, device)
            print(f"  eval step {step}: tok_acc {metrics['tok_acc']:.3f}  seq_acc {metrics['seq_acc']:.3f}  loss {metrics['eval_loss']:.4f}  ppl {metrics['eval_ppl']:.2f}")
            history.append({"step": step, **metrics})
            (out_dir / "history.json").write_text(json.dumps({"history": history, "n_params": n_params}, indent=2))
            if use_wandb:
                import wandb
                wandb.log({f"eval/{k}": v for k, v in metrics.items()} | {"step": step})

    # final save
    ckpt_path = out_dir / "model.pt"
    torch.save({"model": (model.state_dict() if not cfg.compile else model._orig_mod.state_dict()),
                "cfg": asdict(cfg), "model_cfg": asdict(model_cfg),
                "history": history, "train_log": train_log, "n_params": n_params}, ckpt_path)
    (out_dir / "history.json").write_text(json.dumps({"history": history, "n_params": n_params}, indent=2))
    (out_dir / "train_log.json").write_text(json.dumps({"train_log": train_log, "n_params": n_params}, indent=2))
    print(f"[{run_name}] saved -> {ckpt_path}")
    if use_wandb:
        import wandb
        wandb.finish()
    return history
