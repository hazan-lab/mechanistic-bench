"""Evaluator factory for language-modeling downstream / perplexity eval.

Ported from ``olmo.eval.__init__`` with imports redirected to
``mechbench.configs.lm_config`` / ``mechbench.tokenizer`` and the
distributed sampler replaced with plain shuffle=False for single-GPU
operation.
"""

from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, Metric

from ..configs.lm_config import EvaluatorConfig, EvaluatorType, LMTrainConfig
from ..exceptions import OLMoConfigurationError
from ..tokenizer import Tokenizer
from .downstream import ICLMetric, label_to_task_map
from .evaluator import Evaluator

__all__ = [
    "Evaluator",
    "ICLMetric",
    "label_to_task_map",
    "build_downstream_evaluator",
    "build_evaluator",
    "build_evaluators",
]


def _load_tokenizer(cfg: LMTrainConfig) -> Tokenizer:
    return Tokenizer.from_identifier(
        cfg.tokenizer.identifier,
        eos_token_id=cfg.model.eos_token_id,
        pad_token_id=cfg.model.pad_token_id,
        vocab_size=cfg.model.vocab_size,
        truncate_direction=cfg.tokenizer.truncate_direction,
    )


def build_downstream_evaluator(
    train_config: LMTrainConfig,
    eval_cfg: EvaluatorConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    is_unit_test: bool = False,
) -> Evaluator:
    task_kwargs: Dict = {}
    task_class = label_to_task_map[eval_cfg.label]
    if isinstance(task_class, tuple):
        task_class, task_kwargs = task_class
    ds_eval_dataset = task_class(tokenizer=tokenizer, **task_kwargs)  # type: ignore
    data_config = eval_cfg.data
    ds_eval_dataloader = DataLoader(
        ds_eval_dataset,
        batch_size=eval_cfg.device_eval_batch_size or train_config.device_eval_batch_size,
        collate_fn=ds_eval_dataset.collate_fn,
        num_workers=data_config.num_workers,
        shuffle=False,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor,
        persistent_workers=data_config.persistent_workers,
        timeout=data_config.timeout,
    )
    metric = ICLMetric(metric_type=ds_eval_dataset.metric_type)

    evaluator = Evaluator(
        label=eval_cfg.label,
        type=eval_cfg.type,
        eval_loader=ds_eval_dataloader,
        eval_metric=metric.to(device),
        subset_num_batches=eval_cfg.subset_num_batches,
    )
    return evaluator


def build_evaluator(
    train_config: LMTrainConfig,
    eval_config: EvaluatorConfig,
    tokenizer: Tokenizer,
    device: torch.device,
) -> Evaluator:
    from ..data import build_eval_dataloader

    if eval_config.type == EvaluatorType.downstream:
        return build_downstream_evaluator(train_config, eval_config, tokenizer, device)
    elif eval_config.type == EvaluatorType.lm:
        eval_loader = build_eval_dataloader(
            train_config,
            eval_config.data,
            eval_config.device_eval_batch_size or train_config.device_eval_batch_size,
        )

        def make_metric():
            return MeanMetric(nan_strategy="error").to(device)

        eval_metric: Union[Metric, Dict[str, Metric]]
        if eval_config.data.paths:
            eval_metric = make_metric()
        elif eval_config.data.datasets:
            eval_metric = {label: make_metric() for label in eval_config.data.datasets.keys()}
        else:
            raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")

        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    else:
        raise ValueError(f"Unexpected evaluator type '{eval_config.type}'")


def build_evaluators(cfg: LMTrainConfig, device: torch.device) -> List[Evaluator]:
    evaluators = []
    tokenizer = _load_tokenizer(cfg)
    for eval_cfg in cfg.evaluators:
        evaluators.append(build_evaluator(cfg, eval_cfg, tokenizer, device))
    return evaluators
