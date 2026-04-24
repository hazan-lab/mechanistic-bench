# 50M LM training: seq_len=2048 vs seq_len=4096

Both setups Chinchilla-budgeted at 1B tokens (50M params × 20). The
2048 runs use 954 optimizer steps (`global_batch=512 × seq=2048 × 954 ≈
1.0B`); the 4096 runs use 477 optimizer steps with the same global
batch (`global_batch=512 × seq=4096 × 477 ≈ 1.0B`). Identical model
architecture, optimizer, scheduler, and data per arch — the only
deltas are `max_seq_len` (2048→4096), `device_train_microbatch_size`
(16→8 to keep per-device memory similar), `max_duration` (954→477),
and proportionally scaled warmup/eval/save intervals.

| arch | seq | params | c4 ppl ↓ | wt_103 ppl ↓ | piqa ↑ | hellaswag ↑ |
|---|---|---|---|---|---|---|
| attn            | 2048 | 50.29M | **155.5** | **477.8** | 0.536 | 0.255 |
| attn            | 4096 | 50.29M | 288.9     | 929.2     | 0.523 | **0.258** |
| mamba2          | 2048 | 49.70M | **150.6** | **479.8** | **0.546** | **0.255** |
| mamba2          | 4096 | 49.70M | 236.4     | 765.6     | 0.523 | 0.254 |
| alt_attn_mamba2 | 2048 | 50.66M | **155.2** | **465.1** | **0.547** | 0.255 |
| alt_attn_mamba2 | 4096 | 50.66M | 250.1     | 801.5     | 0.521 | **0.257** |
| headwise_mamba2 | 2048 | 49.77M | **159.8** | **546.0** | 0.539 | 0.254 |
| headwise_mamba2 | 4096 | 49.77M | 255.3     | 884.8     | 0.524 | **0.256** |

## Per-arch deltas (4096 minus 2048)

| arch | Δc4 ppl | Δwt ppl | Δpiqa | Δhellaswag |
|---|---|---|---|---|
| attn            | +133.4 (+86%) | +451.5 (+95%) | -0.013 | +0.003 |
| mamba2          | +85.8  (+57%) | +285.9 (+60%) | -0.023 | -0.001 |
| alt_attn_mamba2 | +94.9  (+61%) | +336.4 (+72%) | -0.026 | +0.002 |
| headwise_mamba2 | +95.5  (+60%) | +338.8 (+62%) | -0.015 | +0.002 |

## Within-seq ranking on c4

| seq | best → worst on c4 |
|---|---|
| 2048 | mamba2 (150.6) < alt (155.2) < attn (155.5) < headwise (159.8) |
| 4096 | mamba2 (236.4) < alt (250.1) < headwise (255.3) < attn (288.9) |

## Takeaways

1. **At fixed Chinchilla budget (1B tokens), seq=4096 is strictly worse on perplexity than seq=2048 for every arch** — the regression is +57–95% on c4 ppl. Mechanistically, doubling `max_seq_len` while holding tokens constant halves the number of optimizer steps (954 → 477), and the runs do not recoup the lost gradient-update count.
2. **Mamba-2 takes the biggest relative hit the smallest at 4096** — the gap between `mamba2` and `attn` on c4 widens from `attn − mamba2 = 4.9 ppl` (+3%) at seq=2048 to `attn − mamba2 = 52.5 ppl` (+22%) at seq=4096. This is consistent with longer context being a relative win for the SSM, even though both archs regressed in absolute terms.
3. **`headwise_mamba2` moves from worst on c4 at seq=2048 to second-worst at seq=4096** — but the absolute gap to attn shrinks dramatically (159.8 vs 288.9 — i.e., headwise becomes *much* better than attn at 4096 in absolute terms even though it's worse than the other Mamba/hybrid setups).
4. **ICL metrics (piqa, hellaswag) barely move**, well within noise at this scale and token budget.

## Implication for matching spectral-transformers

Spectral's 150M ablations use seq_len=4096 *and* `global_batch=1024`,
so they process 4× more tokens per step than our seq=2048 runs and 2×
more than our seq=4096 runs. To make a like-for-like comparison
against spectral while keeping our token budget Chinchilla-optimal,
we'd need to ~2× the global batch at seq=4096 — i.e. give back the
gradient-step count we lost in this experiment. The numbers here
isolate seq_len alone.

## Run/log/checkpoint locations

- Configs: `configs/lm/scale_50m_seq4096/{attn,mamba2,alt_attn_mamba2,headwise_mamba2}.yaml`
- Logs: `/scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/logs/seq4096/50m-seq4096-{arch}.log`
- Final checkpoints: `/scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-50m-seq4096-{arch}/step477/`
- 150M-seq4096 configs are also added (`configs/lm/scale_150m_seq4096/{attn,mamba2,alt_attn_mamba2}.yaml`) but **not yet trained** — saved for a future sweep.
