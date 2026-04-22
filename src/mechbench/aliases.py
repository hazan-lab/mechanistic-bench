"""Common type aliases (ported from spectral-transformers `olmo.aliases`)."""

from __future__ import annotations

from os import PathLike
from typing import Union

__all__ = ["PathOrStr"]


PathOrStr = Union[str, PathLike]
