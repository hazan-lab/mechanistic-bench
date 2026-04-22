"""Simple right-padding collator.

Minimal port of ``olmo.data.collator.DataCollator``. Only handles the fields
the single-GPU LM trainer actually needs: ``input_ids``, ``attention_mask``,
``index``, ``metadata``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F

from ..configs.lm_config import PaddingDirection

__all__ = ["DataCollator"]


@dataclass
class DataCollator:
    pad_direction: PaddingDirection = PaddingDirection.right
    pad_token_id: int = 0

    def __call__(self, items: Union[List[Dict[str, Any]], List[torch.Tensor]]) -> Dict[str, Any]:
        assert items
        max_len = max((len(x["input_ids"] if isinstance(x, dict) else x) for x in items))
        all_input_ids = []
        all_attention_mask = []
        all_indices = []
        all_metadata: List[Any] = []

        for x in items:
            input_ids = x["input_ids"] if isinstance(x, dict) else x
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)

            pad_shape = (
                (max_len - len(input_ids), 0)
                if self.pad_direction == PaddingDirection.left
                else (0, max_len - len(input_ids))
            )

            all_input_ids.append(
                F.pad(input_ids.to(dtype=torch.long), pad_shape, value=self.pad_token_id)
            )

            attention_mask = x.get("attention_mask") if isinstance(x, dict) else None
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)
                all_attention_mask.append(
                    F.pad(attention_mask.to(dtype=torch.float), pad_shape, value=0.0)
                )

            index = x.get("index") if isinstance(x, dict) else None
            if index is not None:
                all_indices.append(torch.tensor(index))

            metadata = x.get("metadata") if isinstance(x, dict) else None
            if metadata is not None:
                all_metadata.append(metadata)

        out: Dict[str, Any] = {"input_ids": torch.stack(all_input_ids)}
        if all_attention_mask:
            out["attention_mask"] = torch.stack(all_attention_mask)
        if all_indices:
            out["index"] = torch.stack(all_indices)
        if all_metadata:
            out["metadata"] = all_metadata

        return out
