"""Fixture-resolution shim for ported eval code.

The downstream/eval modules reference path-like helpers formerly located in
``olmo_data``. We keep the module name to minimize diffs against upstream
and redirect file lookups at the ``mechbench.olmo_data`` package directory.
"""

from .data import get_data_path, is_data_dir, is_data_file

__all__ = ["get_data_path", "is_data_dir", "is_data_file"]
