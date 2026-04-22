"""Tests for the LM data pipeline (memmap dataset + collator)."""

from __future__ import annotations

import numpy as np
import torch

from mechbench.data.collator import DataCollator
from mechbench.data.memmap_dataset import MemMapDataset


def _make_memmap(tmp_path, n_tokens: int, vocab_size: int = 100, seed: int = 0):
    path = tmp_path / "toy.npy"
    # Write raw uint16 tokens to file (not .npy metadata; matches olmo's format).
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    return path


def test_memmap_chunks_correctly(tmp_path):
    chunk_size = 8
    n_tokens = 40  # 5 chunks
    path = _make_memmap(tmp_path, n_tokens)

    ds = MemMapDataset(path, chunk_size=chunk_size, include_instance_metadata=True)
    assert len(ds) == 5

    item = ds[0]
    assert "input_ids" in item
    assert item["input_ids"].shape == (chunk_size,)
    assert item["input_ids"].dtype == torch.long
    assert "metadata" in item


def test_memmap_ragged_trims(tmp_path):
    chunk_size = 8
    n_tokens = 45  # 5 chunks + 5 leftover ignored
    path = _make_memmap(tmp_path, n_tokens)

    ds = MemMapDataset(path, chunk_size=chunk_size)
    assert len(ds) == 5


def test_memmap_multi_path(tmp_path):
    chunk_size = 4
    p1 = _make_memmap(tmp_path, 16, seed=1)
    p2 = tmp_path / "other.npy"
    arr = np.arange(20, dtype=np.uint16)
    with open(p2, "wb") as f:
        f.write(arr.tobytes())

    ds = MemMapDataset(p1, p2, chunk_size=chunk_size)
    # 16 // 4 + 20 // 4 = 4 + 5 = 9
    assert len(ds) == 9
    last = ds[8]["input_ids"]
    # last chunk corresponds to tokens [16, 17, 18, 19] in the second file
    assert torch.equal(last, torch.tensor([16, 17, 18, 19], dtype=torch.long))


def test_collator_pads_right():
    coll = DataCollator(pad_token_id=0)
    a = {"input_ids": torch.tensor([1, 2, 3])}
    b = {"input_ids": torch.tensor([4, 5])}
    out = coll([a, b])
    assert out["input_ids"].shape == (2, 3)
    assert torch.equal(out["input_ids"][1], torch.tensor([4, 5, 0]))
