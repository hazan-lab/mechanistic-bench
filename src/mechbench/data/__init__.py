"""Data pipeline for LM pretraining.

Exports memmap dataset / collator / iterable dataset and the
``build_train_dataloader`` / ``build_eval_dataloader`` factory pair that
mirror ``olmo.data`` (simplified for single-GPU operation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader

from ..configs.lm_config import DataConfig, LMTrainConfig
from ..exceptions import OLMoConfigurationError
from .collator import DataCollator
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset

__all__ = [
    "MemMapDataset",
    "DataCollator",
    "IterableDataset",
    "build_memmap_dataset",
    "build_train_dataloader",
    "build_eval_dataloader",
]


def build_memmap_dataset(
    train_config: LMTrainConfig,
    data_config: DataConfig,
    include_instance_metadata: bool = True,
) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []
    if data_config.paths:
        if data_config.datasets:
            raise OLMoConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = list(data_config.paths)
        for path in paths:
            metadata.append({"path": str(path)})
    elif data_config.datasets:
        paths = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    return MemMapDataset(
        *paths,
        chunk_size=train_config.model.max_seq_len,
        memmap_dtype=data_config.effective_memmap_dtype,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=train_config.model.pad_token_id,
        eos_token_id=train_config.model.eos_token_id,
        generate_attention_mask=data_config.generate_attention_mask,
    )


def build_collator(train_config: LMTrainConfig) -> DataCollator:
    return DataCollator(
        pad_direction=train_config.data.pad_direction,
        pad_token_id=train_config.model.pad_token_id,
    )


def build_eval_dataloader(
    train_config: LMTrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = build_memmap_dataset(train_config, data_config, include_instance_metadata=True)
    collator = DataCollator(
        pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id
    )
    if data_config.drop_last:
        assert len(dataset) > 0, f"dataset for {data_config.paths} is empty"
        batch_size = min(batch_size, len(dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        shuffle=shuffle,
        drop_last=data_config.drop_last,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_train_dataloader(
    train_config: LMTrainConfig,
    *,
    include_instance_metadata: bool = False,
) -> DataLoader:
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    collator = build_collator(train_config)
    dataset = build_memmap_dataset(
        train_config, train_config.data, include_instance_metadata=include_instance_metadata
    )
    work_dir: Optional[Path] = None
    if train_config.save_folder:
        work_dir = Path(train_config.save_folder) / "train_data"
        if work_dir.is_dir() and not train_config.save_overwrite:
            # Don't error - allow reusing the cached shuffle. spectral errors
            # here but we're more permissive to make rapid smoke tests easier.
            pass
        work_dir.mkdir(exist_ok=True, parents=True)

    iter_dataset = IterableDataset(
        dataset,
        train_config.global_train_batch_size,
        seed=seed,
        epoch=train_config.epoch or 0,
        shuffle=True,
        drop_last=train_config.data.drop_last,
        world_size=1,
        rank=0,
        fs_local_rank=0,
        work_dir=work_dir,
    )
    return DataLoader(
        iter_dataset,
        batch_size=train_config.device_train_microbatch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
