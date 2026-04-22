"""Smoke test for the ported downstream-eval task fixtures.

Builds one ICL task dataset (PIQA) and confirms that:
  - the class is registered in ``label_to_task_map``
  - a sample can be constructed
  - ``collate_fn`` produces a well-shaped batch

Skips gracefully if the OE eval fixture or the tokenizer file is missing.
"""

from __future__ import annotations

import pytest

from mechbench.eval.downstream import label_to_task_map
from mechbench.olmo_data import is_data_dir, is_data_file
from mechbench.tokenizer import Tokenizer

TOKENIZER_FIXTURE = "tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"
PIQA_FIXTURE = "hf_datasets/piqa"


@pytest.mark.skipif(
    not (is_data_file(TOKENIZER_FIXTURE) and is_data_dir(PIQA_FIXTURE)),
    reason="downstream fixtures missing (olmo_data symlinks not populated)",
)
def test_piqa_task_builds():
    tok = Tokenizer.from_identifier(TOKENIZER_FIXTURE, eos_token_id=0, pad_token_id=1)
    task_entry = label_to_task_map["piqa"]
    task_kwargs = {}
    if isinstance(task_entry, tuple):
        task_class, task_kwargs = task_entry
    else:
        task_class = task_entry
    dataset = task_class(tokenizer=tok, **task_kwargs)

    assert len(dataset) > 0, "PIQA task produced zero samples"
    assert dataset.metric_type in {"acc", "len_norm", "pmi_dc", "ce_loss", "bpb", "f1"}

    # Build one batch via the task's collate_fn.
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert "input_ids" in batch
    assert batch["input_ids"].dim() == 2
    assert batch["input_ids"].size(0) == 2
