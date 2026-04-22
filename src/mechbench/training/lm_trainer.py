"""Single-GPU language-modeling trainer for ``MechModel``.

TODO: FSDP / DDP / torch.compile support is intentionally not ported. This
trainer is single-GPU only; distributed training can be added later once the
eval scoring path is validated end-to-end.

Mirrors the structure of ``training.trainer.train_loop`` but operates on
real text (via the ``data/`` pipeline) and calls the ICL/LM evaluators
ported from spectral-transformers.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, Metric

from ..configs.lm_config import LMTrainConfig, OptimizerType, SchedulerType
from ..data import build_train_dataloader
from ..eval import Evaluator, build_evaluators
from ..eval.downstream import ICLMetric
from ..models.model import MechModel, build_model
from .schedule import cosine_with_warmup

log = logging.getLogger(__name__)


def _precision_dtype(precision: str) -> Optional[torch.dtype]:
    p = precision.lower()
    if p in {"fp32", "amp_fp32", "float32"}:
        return None  # no autocast
    if p in {"amp_bf16", "bf16", "bfloat16"}:
        return torch.bfloat16
    if p in {"amp_fp16", "fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unknown precision '{precision}'")


def _move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out


def _build_optimizer(cfg: LMTrainConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer.name != OptimizerType.adamw:
        raise ValueError(f"unsupported optimizer {cfg.optimizer.name}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )


def _build_scheduler(cfg: LMTrainConfig, optim: torch.optim.Optimizer):
    if cfg.scheduler.name != SchedulerType.cosine_with_warmup:
        if cfg.scheduler.name == SchedulerType.constant:
            # Use cosine with alpha_f=1 to get a constant schedule.
            return cosine_with_warmup(optim, int(cfg.scheduler.t_warmup), cfg.max_steps, min_lr_ratio=1.0)
        raise ValueError(f"unsupported scheduler {cfg.scheduler.name}")
    return cosine_with_warmup(
        optim,
        int(cfg.scheduler.t_warmup),
        cfg.max_steps,
        min_lr_ratio=cfg.scheduler.alpha_f,
    )


def _infinite(loader: DataLoader) -> Iterator[Dict[str, Any]]:
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def run_evaluators(
    model: MechModel,
    evaluators: List[Evaluator],
    device: torch.device,
    precision_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=precision_dtype)
        if precision_dtype is not None
        else _nullcontext()
    )
    for evaluator in evaluators:
        evaluator.reset_metrics()
        n_batches = evaluator.subset_num_batches
        loader = evaluator.eval_loader
        for step_i, batch in enumerate(loader):
            if n_batches is not None and n_batches >= 0 and step_i >= n_batches:
                break
            batch = _move_to_device(batch, device)
            with autocast_ctx:
                logits = model(batch["input_ids"])
            # Compute per-instance CE loss for LM evaluators.
            if isinstance(evaluator.eval_metric, ICLMetric):
                # Downstream (ICL) path — update metric from logits + batch.
                evaluator.update_metrics(batch, torch.tensor(0.0, device=device), logits)
            else:
                # LM path — compute per-row CE loss over the shifted targets.
                logits_shift = logits[:, :-1, :].contiguous()
                targets = batch["input_ids"][:, 1:].contiguous()
                ce = F.cross_entropy(
                    logits_shift.reshape(-1, logits_shift.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                )
                ce = ce.view(targets.size(0), targets.size(1)).mean(dim=1)
                evaluator.update_metrics(batch, ce, logits)
        metrics.update(evaluator.compute_metrics())
    model.train()
    return metrics


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _save_checkpoint(cfg: LMTrainConfig, model: MechModel, optim: torch.optim.Optimizer, step: int, out_dir: Path):
    ckpt_dir = out_dir / f"step{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
            "cfg": dataclasses.asdict(cfg),
        },
        ckpt_dir / "model.pt",
    )
    log.info("saved checkpoint -> %s", ckpt_dir)
    if cfg.save_num_checkpoints_to_keep is not None and cfg.save_num_checkpoints_to_keep > 0:
        _prune_checkpoints(out_dir, cfg.save_num_checkpoints_to_keep)


def _prune_checkpoints(out_dir: Path, keep: int) -> None:
    ckpts = sorted(
        [p for p in out_dir.glob("step*") if p.is_dir()],
        key=lambda p: int(p.name.replace("step", "")),
    )
    if len(ckpts) <= keep:
        return
    for old in ckpts[:-keep]:
        for f in old.glob("*"):
            f.unlink(missing_ok=True)  # type: ignore[arg-type]
        old.rmdir()


def train(cfg: LMTrainConfig) -> Dict[str, Any]:
    """Run the language-modeling trainer.

    Returns a summary dict (history, final step, checkpoint path).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run_name = cfg.run_name or "lm-run"
    save_root = Path(cfg.save_folder)
    save_root.mkdir(parents=True, exist_ok=True)

    # ---- build model ----
    mech_cfg = cfg.model.to_mech_config()
    model = build_model(mech_cfg).to(device)
    n_params = model.num_parameters()
    log.info(
        "[%s] model=%s d_model=%d layers=%d params=%.2fM",
        run_name,
        mech_cfg.block_types,
        mech_cfg.d_model,
        len(mech_cfg.block_types),
        n_params / 1e6,
    )

    # ---- data ----
    train_loader = build_train_dataloader(cfg)
    train_iter = _infinite(train_loader)

    # ---- optimizer + scheduler ----
    optim = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optim)
    precision_dtype = _precision_dtype(cfg.precision)

    # ---- evaluators ----
    evaluators: List[Evaluator] = []
    if cfg.evaluators:
        evaluators = build_evaluators(cfg, device)

    # ---- optional wandb ----
    use_wandb = False
    wandb_mod = None
    if cfg.wandb is not None:
        try:
            import wandb as wandb_mod  # type: ignore
            wandb_mod.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                group=cfg.wandb.group,
                name=cfg.wandb.name or run_name,
                tags=cfg.wandb.tags,
                config={
                    "train": dataclasses.asdict(cfg),
                    "n_params": n_params,
                },
            )
            use_wandb = True
        except Exception as e:  # pragma: no cover
            log.warning("wandb unavailable: %s", e)

    # ---- load checkpoint? ----
    start_step = 0
    if cfg.load_path:
        state = torch.load(cfg.load_path, map_location=device)
        model.load_state_dict(state["model"])
        if "optim" in state:
            optim.load_state_dict(state["optim"])
        start_step = state.get("step", 0)
        log.info("resumed from %s at step %d", cfg.load_path, start_step)

    history: List[Dict[str, Any]] = []
    grad_accum = cfg.device_train_grad_accum
    max_grad_norm = cfg.max_grad_norm

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=precision_dtype)
        if precision_dtype is not None
        else _nullcontext()
    )

    log.info(
        "training for %d steps, global_batch=%d micro=%d accum=%d precision=%s",
        cfg.max_steps,
        cfg.global_train_batch_size,
        cfg.device_train_microbatch_size,
        grad_accum,
        cfg.precision,
    )

    model.train()
    step = start_step
    log_loss = 0.0
    log_n = 0
    t0 = time.time()
    while step < cfg.max_steps:
        optim.zero_grad(set_to_none=True)
        micro_loss_sum = 0.0
        for _ in range(grad_accum):
            micro = next(train_iter)
            micro = _move_to_device(micro, device)
            with autocast_ctx:
                logits = model(micro["input_ids"])
                # Next-token prediction loss over the whole sequence.
                targets = micro["input_ids"][:, 1:].contiguous()
                shifted = logits[:, :-1, :].contiguous()
                loss = F.cross_entropy(
                    shifted.reshape(-1, shifted.size(-1)),
                    targets.reshape(-1),
                )
            (loss / grad_accum).backward()
            micro_loss_sum += loss.detach().item()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optim.step()
        scheduler.step()
        step += 1

        log_loss += micro_loss_sum / grad_accum
        log_n += 1

        if step % cfg.log_interval == 0:
            mean_loss = log_loss / max(1, log_n)
            elapsed = time.time() - t0
            tok_per_s = (
                cfg.global_train_batch_size
                * cfg.model.max_seq_len
                * cfg.log_interval
                / max(1e-9, elapsed)
            )
            lr = scheduler.get_last_lr()[0]
            log.info(
                "step %d/%d  loss %.4f  lr %.2e  tok/s %.0f",
                step,
                cfg.max_steps,
                mean_loss,
                lr,
                tok_per_s,
            )
            if use_wandb and wandb_mod is not None:
                wandb_mod.log(
                    {"train/loss": mean_loss, "train/lr": lr, "step": step},
                    step=step,
                )
            history.append({"step": step, "loss": mean_loss, "lr": lr})
            log_loss = 0.0
            log_n = 0
            t0 = time.time()

        if evaluators and cfg.eval_interval > 0 and step % cfg.eval_interval == 0:
            metrics = run_evaluators(model, evaluators, device, precision_dtype)
            log.info("eval @ step %d: %s", step, metrics)
            history.append({"step": step, **{f"eval/{k}": v for k, v in metrics.items()}})
            if use_wandb and wandb_mod is not None:
                wandb_mod.log({**metrics, "step": step}, step=step)

        if cfg.save_interval and cfg.save_interval > 0 and step % cfg.save_interval == 0:
            _save_checkpoint(cfg, model, optim, step, save_root)

    # Final save and final eval.
    _save_checkpoint(cfg, model, optim, step, save_root)
    if evaluators:
        final_metrics = run_evaluators(model, evaluators, device, precision_dtype)
        log.info("final eval: %s", final_metrics)
        history.append({"step": step, **{f"eval/{k}": v for k, v in final_metrics.items()}})

    (save_root / "history.json").write_text(json.dumps(history, indent=2))
    if use_wandb and wandb_mod is not None:
        wandb_mod.finish()

    return {"history": history, "final_step": step, "save_folder": str(save_root)}
