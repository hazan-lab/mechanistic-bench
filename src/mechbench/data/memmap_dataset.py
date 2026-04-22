"""Memory-mapped uint16 token dataset.

Ported from ``olmo.data.memmap_dataset.MemMapDataset``, with the cloud
(S3/R2/Weka) I/O, instance-filter, label-mask, and generate-doc-lengths
paths removed. Reads plain files via ``numpy.memmap``.
"""

from __future__ import annotations

from copy import deepcopy
from os import PathLike
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..aliases import PathOrStr

__all__ = ["MemMapDataset"]


class MemMapDataset(Dataset[Dict[str, Any]]):
    """
    A PyTorch :class:`~torch.utils.data.Dataset` backed by one or more numpy
    memory-mapped arrays of token IDs. Token IDs are chunked together into
    contiguous blocks of ``chunk_size`` to create instances.

    If the length of a memory-mapped array is not a multiple of ``chunk_size``
    the remainder of the tokens will be ignored.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        chunk_size: int = 1024,
        memmap_dtype: Union[
            Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]
        ] = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: bool = True,
        generate_attention_mask: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        if not paths:
            raise ValueError("At least one path is required")

        if generate_attention_mask and pad_token_id is None:
            raise ValueError("'pad_token_id' is required for 'generate_attention_mask'")

        if isinstance(metadata, list):
            if len(metadata) != len(paths):
                raise ValueError("'metadata' should have the same length as the number of file paths")
        else:
            metadata = [metadata or {}] * len(paths)

        self._memmap_paths = [str(p) for p in paths]
        self._metadata = metadata
        self._chunk_size = chunk_size
        self._mmap_offsets: Optional[List[Tuple[int, int]]] = None
        self._num_instances: Optional[int] = None
        self.dtype = memmap_dtype
        self._include_instance_metadata = include_instance_metadata
        self._generate_attention_mask = generate_attention_mask
        self._pad_token_id = pad_token_id
        self._eos_token_id = eos_token_id

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def max_seq_len(self) -> int:
        return self.chunk_size

    def _file_num_chunks(self, path: str) -> int:
        import os

        item_size = np.dtype(self.dtype).itemsize
        file_bytes = os.path.getsize(path)
        return file_bytes // (item_size * self._chunk_size)

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        if self._mmap_offsets is None:
            self._mmap_offsets = []
            start_offset = 0
            for path in self._memmap_paths:
                length = self._file_num_chunks(path)
                end_offset = start_offset + length
                self._mmap_offsets.append((start_offset, end_offset))
                start_offset += length
        return self._mmap_offsets

    def _read_chunk_from_memmap(self, path: str, index: int) -> torch.Tensor:
        item_size = np.dtype(self.dtype).itemsize
        byte_start = index * item_size * self._chunk_size
        # Use numpy memmap to read a window of ``chunk_size`` elements.
        array = np.memmap(
            path,
            dtype=self.dtype,
            mode="r",
            offset=byte_start,
            shape=(self._chunk_size,),
        )
        # Copy into a regular int64 tensor so downstream code doesn't hold
        # onto the memmap.
        return torch.from_numpy(np.asarray(array).astype(np.int64))

    def __len__(self) -> int:
        if self._num_instances is None:
            self._num_instances = self.offsets[-1][1]
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        pos_index = index if index >= 0 else len(self) + index

        memmap_index: Optional[int] = None
        memmap_local_index: Optional[int] = None
        for i, (offset_start, offset_end) in enumerate(self.offsets):
            if offset_start <= pos_index < offset_end:
                memmap_index = i
                memmap_local_index = pos_index - offset_start
                break

        if memmap_index is None or memmap_local_index is None:
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        input_ids = self._read_chunk_from_memmap(self._memmap_paths[memmap_index], memmap_local_index)
        out: Dict[str, Any] = {"input_ids": input_ids}

        if self._include_instance_metadata:
            out["metadata"] = deepcopy(self._metadata[memmap_index])

        if self._generate_attention_mask:
            assert self._pad_token_id is not None
            attn_mask = torch.ones_like(input_ids)
            attn_mask.masked_fill_(input_ids == self._pad_token_id, 0)
            out["attention_mask"] = attn_mask

        return out

    def __add__(self, other: "MemMapDataset") -> "MemMapDataset":
        if not isinstance(other, MemMapDataset):
            raise NotImplementedError(f"Expected another MemMapDataset but got {type(other)}")
        return MemMapDataset(
            *(self._memmap_paths + other._memmap_paths),
            chunk_size=self._chunk_size,
            memmap_dtype=self.dtype,
            metadata=list(self._metadata) + list(other._metadata),
        )
