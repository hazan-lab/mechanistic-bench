"""Selected utility functions ported from ``olmo.util``.

Only the helpers that the language-modeling data loaders, tokenizer, and
downstream-eval code actually depend on are included. The S3/logging/
cached-path kitchen sink is deliberately dropped.
"""

from __future__ import annotations

import gzip
import json
import os
from enum import Enum
from itertools import cycle, islice
from queue import Queue
from threading import Thread
from typing import Optional

import datasets

from .exceptions import OLMoThreadError

__all__ = [
    "StrEnum",
    "load_hf_dataset",
    "load_oe_eval_requests",
    "threaded_generator",
    "roundrobin",
    "default_thread_count",
]


class StrEnum(str, Enum):
    """
    Equivalent to Python's :class:`enum.StrEnum` (added in 3.11). Kept as a
    local copy so that the downstream/eval modules and config schemas can
    import it without pulling in extra machinery.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


def _get_data_path(rel_path: str):
    """Resolve a path within the bundled ``olmo_data`` package directory.

    Mirrors the behavior of ``olmo_data.get_data_path`` but returns a plain
    :class:`pathlib.Path` rather than a context manager since all our data is
    on a real filesystem.
    """
    from .olmo_data import get_data_path

    return get_data_path(rel_path)


def load_hf_dataset(path: str, name: Optional[str], split: str):
    """
    Loads a HuggingFace dataset cached under ``olmo_data/hf_datasets``.
    """
    dataset_rel_path = os.path.join("hf_datasets", path, name or "none", split)
    with _get_data_path(dataset_rel_path) as dataset_path:
        if not dataset_path.is_dir():
            raise NotADirectoryError(
                f"HF dataset {path} name {name} split {split} not found in directory {dataset_rel_path}"
            )
        return datasets.load_from_disk(str(dataset_path))


def load_oe_eval_requests(path: str, name: Optional[str] = None, split: Optional[str] = None):
    """
    Loads an oe-eval request file from ``olmo_data/oe_eval_tasks``.
    """
    dataset_rel_path = os.path.join("oe_eval_tasks", path)
    if name is not None:
        dataset_rel_path = os.path.join(dataset_rel_path, name)
    with _get_data_path(dataset_rel_path) as dataset_path:
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"OE Eval dataset not found in directory {dataset_rel_path}")
        data_file = dataset_path / "requests.jsonl.gz"
        if not data_file.is_file():
            data_file = dataset_path / "requests.jsonl"
        if not data_file.is_file():
            raise FileNotFoundError(
                f"OE Eval dataset file requests-{split}.jsonl(.gz) missing in directory {dataset_rel_path}"
            )
        requests = []
        if data_file.suffix == ".gz":
            with gzip.open(data_file, "r") as file:
                for line in file:
                    requests.append(json.loads(line.decode("utf-8").strip()))
        else:
            with open(data_file, "r") as file:
                for line2 in file:
                    requests.append(json.loads(line2.strip()))
        config = None
        config_file = dataset_path / "config.json"
        if config_file.is_file():
            with open(config_file, "r") as file:
                config = json.load(file)
        return config, requests


def default_thread_count() -> int:
    return int(os.environ.get("OLMO_NUM_THREADS") or min(32, (os.cpu_count() or 1) + 4))


def threaded_generator(g, maxsize: int = 16, thread_name: Optional[str] = None):
    q: Queue = Queue(maxsize=maxsize)
    sentinel = object()

    def fill_queue():
        try:
            for value in g:
                q.put(value)
        except Exception as e:
            q.put(e)
        finally:
            q.put(sentinel)

    thread_name = thread_name or repr(g)
    thread = Thread(name=thread_name, target=fill_queue, daemon=True)
    thread.start()

    for x in iter(q.get, sentinel):
        if isinstance(x, Exception):
            raise OLMoThreadError(f"generator thread {thread_name} failed") from x
        else:
            yield x


def roundrobin(*iterables):
    """
    Call the given iterables in a round-robin fashion. For example:
    ``roundrobin('ABC', 'D', 'EF') --> A D E B F C``
    """
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))
