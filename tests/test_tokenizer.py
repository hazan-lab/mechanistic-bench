"""Tokenizer roundtrip smoke test."""

from __future__ import annotations

import pytest

from mechbench.olmo_data import is_data_file
from mechbench.tokenizer import Tokenizer

TOKENIZER_FIXTURE = "tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"


@pytest.mark.skipif(
    not is_data_file(TOKENIZER_FIXTURE),
    reason="GPT-NeoX tokenizer fixture missing (requires olmo_data symlinks)",
)
def test_tokenizer_roundtrip():
    tok = Tokenizer.from_identifier(TOKENIZER_FIXTURE, eos_token_id=0, pad_token_id=1)
    text = "Hello world! Mechanistic-bench test."
    ids = tok.encode(text, add_special_tokens=False)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids, skip_special_tokens=True)
    # GPT-NeoX tokenizer should decode back to (approximately) the same text.
    assert "Hello" in decoded and "mechanistic" in decoded.lower()


@pytest.mark.skipif(
    not is_data_file(TOKENIZER_FIXTURE),
    reason="tokenizer fixture missing",
)
def test_tokenizer_add_special_tokens_appends_eos():
    tok = Tokenizer.from_identifier(TOKENIZER_FIXTURE, eos_token_id=0, pad_token_id=1)
    ids = tok.encode("hi", add_special_tokens=True)
    assert ids[-1] == 0
