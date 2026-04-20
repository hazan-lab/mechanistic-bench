"""OmegaConf/dataclass configs for language-modeling training.

Ported/simplified from ``olmo.config``. We keep the field names and YAML
layout compatible with spectral-transformers configs where possible, so
that their pretraining YAMLs can be consumed by swapping paths and the
``model`` block.

Only the pieces needed for a single-GPU, single-node LM trainer are
included. In particular we drop FSDP/DDP, sharded checkpointer variants,
speed monitor, and the compiler config wrapper.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException

from ..aliases import PathOrStr
from ..exceptions import OLMoConfigurationError
from ..models.model import MechConfig
from ..util import StrEnum

__all__ = [
    "OptimizerType",
    "OptimizerConfig",
    "SchedulerType",
    "SchedulerConfig",
    "PaddingDirection",
    "DataConfig",
    "EvaluatorType",
    "EvaluatorConfig",
    "TokenizerConfig",
    "WandbConfig",
    "LMModelConfig",
    "LMTrainConfig",
]

C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


class BaseConfig:
    """Minimal BaseConfig that supports YAML loading + dotlist overrides."""

    @classmethod
    def _register_resolvers(cls, validate_paths: bool = True):
        def path_glob(*paths) -> List[str]:
            out = []
            for path in paths:
                matches = sorted(glob(path))
                if not matches and validate_paths:
                    raise FileNotFoundError(f"{path} does not match any files or dirs")
                out.extend(matches)
            return out

        def path_choose(*paths) -> str:
            for path in paths:
                if Path(path).exists():
                    return path
            if validate_paths:
                raise FileNotFoundError(", ".join(paths))
            else:
                return ""

        om.register_new_resolver("path.glob", path_glob, replace=True)
        om.register_new_resolver("path.choose", path_choose, replace=True)

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        return config

    @classmethod
    def new(cls: Type[C], **kwargs) -> C:
        cls._register_resolvers()
        conf = om.structured(cls)
        try:
            if kwargs:
                conf = om.merge(conf, kwargs)
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))

    @classmethod
    def load(
        cls: Type[C],
        path: PathOrStr,
        overrides: Optional[List[str]] = None,
        key: Optional[str] = None,
        validate_paths: bool = True,
    ) -> C:
        """Load from a YAML file with optional dotlist overrides."""
        cls._register_resolvers(validate_paths=validate_paths)
        schema = om.structured(cls)
        try:
            raw = om.load(str(path))
            if key is not None:
                raw = raw[key]  # type: ignore
            raw = cls.update_legacy_settings(raw)
            conf = om.merge(schema, raw)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))

    def save(self, path: PathOrStr) -> None:
        om.save(config=self, f=str(path))

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        out = asdict(self)  # type: ignore
        if exclude is not None:
            for name in exclude:
                if name in out:
                    del out[name]
        return out

    def update_with(self, **kwargs):
        result = deepcopy(self)
        for key, value in kwargs.items():
            setattr(result, key, value)
        return result


class OptimizerType(StrEnum):
    adamw = "adamw"


@dataclass
class OptimizerConfig(BaseConfig):
    name: OptimizerType = OptimizerType.adamw
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    metrics_log_interval: Optional[int] = None

    def __post_init__(self):
        self.betas = tuple(self.betas)  # type: ignore[assignment]


class SchedulerType(StrEnum):
    cosine_with_warmup = "cosine_with_warmup"
    constant = "constant"


@dataclass
class SchedulerConfig(BaseConfig):
    name: SchedulerType = SchedulerType.cosine_with_warmup
    t_warmup: Union[int, float] = 100
    t_max: Optional[Union[int, float]] = None
    alpha_f: float = 0.1
    grad_clip_warmup_steps: Optional[Union[int, float]] = None
    grad_clip_warmup_factor: Optional[float] = None
    warmup_min_lr: Optional[float] = None


class PaddingDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class DataConfig(BaseConfig):
    paths: Optional[List[str]] = None
    memmap_dtype: str = "uint16"
    datasets: Optional[Dict[str, List[str]]] = None
    pad_direction: PaddingDirection = PaddingDirection.right
    generate_attention_mask: bool = False
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    timeout: int = 0
    seed: Optional[int] = None

    @property
    def effective_memmap_dtype(self):
        try:
            np.dtype(dtype := getattr(np, self.memmap_dtype))
        except (AttributeError, TypeError) as e:
            raise TypeError(f"Value {self.memmap_dtype} is not a valid numpy type") from e
        return dtype


class EvaluatorType(StrEnum):
    downstream = "downstream"
    lm = "lm"


@dataclass
class EvaluatorConfig(BaseConfig):
    label: str = ""
    type: EvaluatorType = EvaluatorType.lm
    data: DataConfig = field(default_factory=DataConfig)
    device_eval_batch_size: Optional[int] = None
    subset_num_batches: Optional[int] = None


@dataclass
class TokenizerConfig(BaseConfig):
    identifier: str = "tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"
    truncate_direction: str = "right"


@dataclass
class WandbConfig(BaseConfig):
    project: Optional[str] = None
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    log_artifacts: bool = False
    rank_zero_only: bool = True
    log_interval: int = 1


# -------- Model config wrapper --------
# We mirror ``MechConfig`` fields with YAML-friendly defaults so the
# OmegaConf schema can round-trip them. We keep the ``block_types`` list and
# add a couple of aliases (``vocab_size`` / ``max_sequence_length``) for
# compatibility with spectral YAMLs.


@dataclass
class LMModelConfig(BaseConfig):
    """Subset of :class:`MechConfig` fields we surface in YAML.

    This intentionally duplicates the MechConfig dataclass because OmegaConf's
    ``structured`` API needs default factories for lists and the conversion
    back to a plain ``MechConfig`` is trivial.
    """

    vocab_size: int = 50304
    d_model: int = 320
    n_heads: int = 8
    max_seq_len: int = 1024
    mlp_hidden_mult: float = 2.0
    block_types: List[str] = field(default_factory=lambda: ["attn"] * 8)
    # mamba hyperparams
    d_state: int = 16
    d_conv: int = 4
    mamba_expand: int = 2
    # attention opts
    rope: bool = True
    use_flash: bool = True
    # headwise hybrid
    n_attn_heads: int = 2
    # regularization / embedding
    dropout: float = 0.0
    tie_embeddings: bool = True
    # tokens used by collator / tokenizer; not consumed by MechConfig itself
    pad_token_id: int = 1
    eos_token_id: int = 0

    def to_mech_config(self) -> MechConfig:
        return MechConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            max_seq_len=self.max_seq_len,
            mlp_hidden_mult=self.mlp_hidden_mult,
            block_types=list(self.block_types),
            d_state=self.d_state,
            d_conv=self.d_conv,
            mamba_expand=self.mamba_expand,
            rope=self.rope,
            use_flash=self.use_flash,
            n_attn_heads=self.n_attn_heads,
            dropout=self.dropout,
            tie_embeddings=self.tie_embeddings,
        )


@dataclass
class LMTrainConfig(BaseConfig):
    """Top-level LM training configuration.

    Closely mirrors ``olmo.config.TrainConfig``, minus distributed training,
    sharded checkpointers, and other OLMo-specific bells and whistles.
    """

    run_name: Optional[str] = None
    seed: int = 6198
    dry_run: bool = False
    epoch: Optional[int] = None

    model: LMModelConfig = field(default_factory=LMModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    evaluators: List[EvaluatorConfig] = field(default_factory=list)
    eval_interval: int = 1000

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    save_folder: str = "./runs"
    save_interval: int = 1000
    save_num_checkpoints_to_keep: int = -1
    save_overwrite: bool = False
    load_path: Optional[str] = None

    max_duration: Union[int, str] = 1000
    global_train_batch_size: int = 64
    device_train_microbatch_size: int = 16
    device_eval_batch_size: int = 16
    eval_subset_num_batches: int = -1

    max_grad_norm: Optional[float] = 1.0

    precision: str = "amp_bf16"

    wandb: Optional[WandbConfig] = None

    log_interval: int = 10

    @property
    def device_train_batch_size(self) -> int:
        """Number of instances per device per optimizer step."""
        return self.global_train_batch_size  # single-GPU only for now

    @property
    def device_train_grad_accum(self) -> int:
        assert self.device_train_microbatch_size > 0
        assert self.device_train_batch_size % self.device_train_microbatch_size == 0, (
            f"device_train_microbatch_size ({self.device_train_microbatch_size}) must divide "
            f"device_train_batch_size ({self.device_train_batch_size})"
        )
        return self.device_train_batch_size // self.device_train_microbatch_size

    @property
    def max_steps(self) -> int:
        if isinstance(self.max_duration, int):
            return self.max_duration
        if isinstance(self.max_duration, str):
            # We don't support token-based durations in this trimmed trainer.
            s = self.max_duration.strip()
            if s.endswith("T") or s.endswith("t"):
                raise OLMoConfigurationError(
                    "token-based max_duration is not supported in mechbench LM trainer; "
                    "please specify steps as a plain integer"
                )
            return int(s)
        raise OLMoConfigurationError(f"max_duration must be int or str, got {type(self.max_duration)}")
