"""Fixture-path helpers used by the ported eval code.

Returns real filesystem paths for resources that live under this package
directory. The canonical ``olmo_data`` version used ``importlib_resources``
which we swap out for a simpler ``pathlib`` approach since our fixtures
(``oe_eval_tasks``, ``hf_datasets``, ``tokenizers``) are just symlinks to
real directories on disk.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

_PACKAGE_DIR = Path(__file__).resolve().parent


def _resolve(data_rel_path: str) -> Path:
    return _PACKAGE_DIR / data_rel_path


def is_data_dir(data_rel_path: str) -> bool:
    return _resolve(data_rel_path).is_dir()


def is_data_file(data_rel_path: str) -> bool:
    return _resolve(data_rel_path).is_file()


@contextmanager
def get_data_path(data_rel_path: str) -> Generator[Path, None, None]:
    try:
        yield _resolve(data_rel_path)
    finally:
        pass
