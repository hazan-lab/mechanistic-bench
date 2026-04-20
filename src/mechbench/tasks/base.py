"""Task base class + vocabulary conventions shared across mechanistic tasks.

Vocabulary layout (matches the JAX spectral-transformer reference):
    PAD=0, BOS=1, SEP=2, QUERY=3, MARKER=4, content tokens 5..vocab_size-1.

Each task yields ``TaskBatch`` dicts with:
    inputs        (B, T)   int64 tokens fed to the model
    targets       (B, T)   int64 next-token labels (shifted; -100 where masked)
    loss_mask     (B, T)   bool  positions that count toward loss & accuracy

The input/target layout follows standard next-token prediction:
    inputs[t]  == records[t]
    targets[t] == records[t+1]
where ``records`` has length (T+1). We then copy ``ignore_index=-100`` into
positions where loss_mask is False so standard cross_entropy skips them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

PAD = 0
BOS = 1
SEP = 2
QUERY = 3
MARKER = 4
CONTENT_LO = 5


@dataclass
class TaskBatch:
    inputs: torch.Tensor        # (B, T) long
    targets: torch.Tensor       # (B, T) long, -100 = ignore
    loss_mask: torch.Tensor     # (B, T) bool


@dataclass
class TaskSpec:
    name: str
    seq_len: int
    vocab_size: int = 64

    @property
    def content_hi(self) -> int:
        return self.vocab_size - 1

    @property
    def n_content(self) -> int:
        return self.vocab_size - CONTENT_LO


class TaskGenerator(Protocol):
    """Callable that produces a ``(records, loss_mask)`` pair of numpy arrays.

    records   : (B, seq_len+1) int32
    loss_mask : (B, seq_len)   bool   (True = count this position)
    """

    def __call__(self, rng: np.random.Generator, batch_size: int, spec: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
        ...


def records_to_batch(records: np.ndarray, loss_mask: np.ndarray, device: torch.device | str = "cpu") -> TaskBatch:
    """Convert (records, loss_mask) numpy arrays to a TaskBatch on ``device``."""
    records_t = torch.from_numpy(records.astype(np.int64))
    inputs = records_t[:, :-1].contiguous()
    targets = records_t[:, 1:].contiguous().clone()
    mask_t = torch.from_numpy(loss_mask.astype(bool))
    targets[~mask_t] = -100
    return TaskBatch(
        inputs=inputs.to(device, non_blocking=True),
        targets=targets.to(device, non_blocking=True),
        loss_mask=mask_t.to(device, non_blocking=True),
    )


def sample_content(rng: np.random.Generator, spec: TaskSpec, shape) -> np.ndarray:
    """Uniform sample over the content-token range [CONTENT_LO, vocab_size)."""
    return rng.integers(CONTENT_LO, spec.vocab_size, size=shape, dtype=np.int32)
