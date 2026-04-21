# 150M-parameter LM configs

Four architectures, param-matched to ~150M (±1%), trained to Chinchilla-optimal
(150M × 20 = 3B tokens) at seq_len 2048.

| arch                      | d_model | mlp_mult | params   | delta |
|---------------------------|--------:|---------:|---------:|------:|
| `attn.yaml`               |     720 |     4.75 | 149.87M  | -0.09% |
| `mamba.yaml`              |     624 |     3.75 | 150.30M  | +0.20% |
| `alt_attn_mamba.yaml`     |     720 |     3.00 | 148.69M  | -0.88% |
| `headwise_alt_attn_mamba` |     720 |     3.50 | 150.24M  | +0.16% |

**Shared training settings**: 12 layers, vocab 50304 (GPT-NeoX padded),
seq_len 2048, global_batch 512 (≈1.05M tokens/step), `max_duration=2862`
steps → 3.00B tokens ≈ Chinchilla optimum. AdamW lr 3e-4 betas (0.9, 0.95),
cosine schedule with 200-step warmup and alpha_f=0.1, grad clip 1.0,
bf16 autocast, `device_train_microbatch_size=8` (→ 64× grad accumulation).

Checkpoints and history land in
`/scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/${run_name}`.

**Evaluators** (same set as `mechbench-10m.yaml`): LM perplexity on
`c4_en` and `wikitext_103` validation memmaps; ICL downstream on `piqa`
and `hellaswag`.

Re-measure param counts with `uv run python scripts/tune_150m.py`.

`headwise_alt_attn_mamba` = alternating `headwise` (head-level attn+mamba)
with `attn`/`mamba` pure layers:
`[headwise, attn, headwise, mamba, headwise, attn, headwise, mamba, ...]`.

Run: `uv run python scripts/train_lm.py configs/lm/150m/<arch>.yaml`
