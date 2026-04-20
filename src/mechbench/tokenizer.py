"""Lightweight wrapper around a HuggingFace ``tokenizers.Tokenizer``.

Ported from ``olmo.tokenizer`` with references to ``TrainConfig`` replaced
by direct parameters. This is used only by eval-time ICL code; training data
is consumed pre-tokenized.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from tokenizers import Tokenizer as BaseTokenizer

from .aliases import PathOrStr
from .exceptions import OLMoConfigurationError
from .olmo_data import get_data_path, is_data_file
from .util import StrEnum

__all__ = ["Tokenizer", "TruncationDirection"]


class TruncationDirection(StrEnum):
    right = "right"
    left = "left"


class Tokenizer:
    """
    A :class:`Tokenizer` is a light-weight wrapper around a HuggingFace
    :class:`tokenizers.Tokenizer`.
    """

    def __init__(
        self,
        base_tokenizer: BaseTokenizer,
        eos_token_id: int,
        pad_token_id: Optional[int] = None,
        truncate_to: Optional[int] = None,
        truncate_direction: Union[str, TruncationDirection] = TruncationDirection.right,
    ):
        self.base_tokenizer = base_tokenizer
        self.base_tokenizer.no_truncation()
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        self.truncate_to = truncate_to
        self.truncate_direction = TruncationDirection(truncate_direction)

    @property
    def vocab_size(self) -> int:
        return self.base_tokenizer.get_vocab_size()

    @property
    def eos_token(self) -> str:
        return self.decode([self.eos_token_id], skip_special_tokens=False)

    @property
    def pad_token(self) -> str:
        return self.decode([self.pad_token_id], skip_special_tokens=False)

    @classmethod
    def from_identifier(
        cls,
        identifier: str,
        *,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        vocab_size: Optional[int] = None,
        truncate_to: Optional[int] = None,
        truncate_direction: Union[str, TruncationDirection] = TruncationDirection.right,
    ) -> "Tokenizer":
        """Resolve a tokenizer identifier (filesystem path, packaged fixture, or HF Hub id)."""
        kwargs = dict(
            pad_token_id=pad_token_id,
            truncate_to=truncate_to,
            truncate_direction=truncate_direction,
        )
        eos_kwargs = {"eos_token_id": eos_token_id} if eos_token_id is not None else {}
        if Path(identifier).is_file():
            tokenizer = cls.from_file(identifier, **eos_kwargs, **kwargs)
        elif is_data_file(identifier):
            with get_data_path(identifier) as tokenizer_path:
                tokenizer = cls.from_file(tokenizer_path, **eos_kwargs, **kwargs)
        else:
            tokenizer = cls.from_pretrained(identifier, **eos_kwargs, **kwargs)
        if vocab_size is not None and vocab_size < tokenizer.vocab_size:
            raise OLMoConfigurationError(
                f"config vocab_size={vocab_size} is smaller than tokenizer vocab_size={tokenizer.vocab_size}"
            )
        return tokenizer

    @classmethod
    def from_pretrained(cls, identifier: str, **kwargs) -> "Tokenizer":
        base_tokenizer = BaseTokenizer.from_pretrained(identifier)
        eos_token_id = kwargs.pop("eos_token_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_token_id, **kwargs)

    @classmethod
    def from_file(cls, filename: PathOrStr, **kwargs) -> "Tokenizer":
        base_tokenizer = BaseTokenizer.from_file(str(filename))
        eos_token_id = kwargs.pop("eos_token_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_token_id, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: PathOrStr) -> "Tokenizer":
        """Load a tokenizer given a path to a directory containing config.yaml."""
        from cached_path import cached_path
        from omegaconf import OmegaConf

        config_path = cached_path(os.path.join(checkpoint_dir, "config.yaml"))
        conf = OmegaConf.load(str(config_path))
        tokenizer_cfg = conf.get("tokenizer", {})
        model_cfg = conf.get("model", {})
        identifier = tokenizer_cfg.get("identifier")
        truncate_direction = tokenizer_cfg.get("truncate_direction", "right")
        eos_token_id = model_cfg.get("eos_token_id")
        pad_token_id = model_cfg.get("pad_token_id")
        vocab_size = model_cfg.get("vocab_size")
        return cls.from_identifier(
            identifier,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            truncate_direction=truncate_direction,
        )

    def add_special_tokens(self, input_ids: List[int]) -> List[int]:
        if not input_ids or input_ids[-1] != self.eos_token_id:
            input_ids.append(self.eos_token_id)
        return input_ids

    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:
        return 2 if is_pair else 1

    def _truncate(
        self,
        input_ids: List[int],
        truncate_to: Optional[int],
        direction: TruncationDirection,
    ) -> List[int]:
        if truncate_to is None or len(input_ids) <= truncate_to:
            return input_ids
        elif direction == TruncationDirection.left:
            return input_ids[len(input_ids) - truncate_to :]
        else:
            return input_ids[: -(len(input_ids) - truncate_to)]

    def encode(self, input: str, add_special_tokens: bool = True) -> List[int]:
        return self.encode_batch([input], add_special_tokens=add_special_tokens)[0]

    def encode_batch(self, inputs: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        truncate_to = self.truncate_to
        if truncate_to is not None and add_special_tokens:
            truncate_to -= self.num_special_tokens_to_add(False)

        batch_encoding = self.base_tokenizer.encode_batch(inputs)

        all_input_ids = []
        for encoding in batch_encoding:
            input_ids = self._truncate(encoding.ids, truncate_to, self.truncate_direction)
            if add_special_tokens:
                input_ids = self.add_special_tokens(input_ids)
            all_input_ids.append(input_ids)

        return all_input_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
