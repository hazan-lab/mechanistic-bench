# 50M LM training: seq_len 2048 vs 4096 vs 8192

This sweep adds `seq_len=8192` configs at the 50M scale to test
the hypothesis that **more context per gradient step** helps SSMs catch
or beat attention. To isolate that effect from the gradient-step regression
seen in the prior seq=4096 sweep, the seq=8192 runs use the **same step
count (477)** as the seq=4096 runs — which means they consume **2× the
tokens** (2 B vs 1 B = 2× Chinchilla). The seq=2048 runs ran 954 steps
(1 B tokens, 1× Chinchilla) for reference. All runs use the same archs,
optimizer, scheduler, data, and `global_batch=512`.

| field | seq=2048 | seq=4096 | seq=8192 |
|---|---|---|---|
| `model.max_seq_len` | 2048 | 4096 | 8192 |
| `device_train_microbatch_size` | 16 | 8 | 4 |
| `max_duration` (steps) | 954 | 477 | 477 |
| total tokens | 1.0 B | 1.0 B | 2.0 B |
| Chinchilla budget | 1× | 1× | 2× |
| optimizer steps | 954 | 477 | **same as 4096** |

## Final evals

| arch | seq | params | c4 ppl ↓ | wt_103 ppl ↓ | piqa ↑ | hellaswag ↑ |
|---|---|---|---|---|---|---|
| attn            | 2048 | 50.29M | **155.5** | **477.8** | **0.536** | 0.255 |
| attn            | 4096 | 50.29M |   288.9   |   929.2   |   0.523   | **0.258** |
| attn            | 8192 | 50.29M |   267.8   |   875.7   |   0.515   | **0.258** |
| mamba2          | 2048 | 49.70M | **150.6** |   479.8   | **0.546** | 0.255 |
| mamba2          | 4096 | 49.70M |   236.4   |   765.6   |   0.523   | 0.254 |
| mamba2          | 8192 | 49.70M |   221.6   | **714.7** |   0.516   | 0.254 |
| alt_attn_mamba2 | 2048 | 50.66M | **155.2** | **465.1** | **0.547** | 0.255 |
| alt_attn_mamba2 | 4096 | 50.66M |   250.1   |   801.5   |   0.521   | **0.257** |
| alt_attn_mamba2 | 8192 | 50.66M |   234.7   |   747.4   |   0.518   | 0.256 |
| headwise_mamba2 | 2048 | 49.77M | **159.8** | **546.0** |   0.539   | 0.254 |
| headwise_mamba2 | 4096 | 49.77M |   255.3   |   884.8   |   0.524   | **0.256** |
| headwise_mamba2 | 8192 | 49.77M |   240.4   |   832.1   | **0.529** | 0.256 |

(Bold = best within-arch across seq lengths.)

## Per-arch deltas

### seq=4096 → seq=8192 (same step count, 2× tokens)

| arch | Δc4 ppl | Δwt ppl | Δpiqa | Δhellaswag |
|---|---|---|---|---|
| attn            | -21.1 (-7.3%) | -53.5 (-5.8%) | -0.008 |  0.000 |
| mamba2          | -14.8 (-6.3%) | -50.9 (-6.6%) | -0.007 |  0.000 |
| alt_attn_mamba2 | -15.4 (-6.2%) | -54.1 (-6.7%) | -0.003 | -0.001 |
| headwise_mamba2 | -14.9 (-5.8%) | -52.7 (-6.0%) |  0.005 |  0.000 |

### seq=8192 vs seq=2048 (still worse despite 2× tokens)

| arch | Δc4 ppl vs 2048 | Δwt ppl vs 2048 |
|---|---|---|
| attn            | +112 (+72%) | +398 (+83%) |
| mamba2          |  +71 (+47%) | +235 (+49%) |
| alt_attn_mamba2 |  +80 (+51%) | +282 (+61%) |
| headwise_mamba2 |  +81 (+50%) | +286 (+52%) |

## Within-seq c4 ranking (lower is better)

| seq | best → worst |
|---|---|
| 2048 | mamba2 (150.6) < alt (155.2) < attn (155.5) < headwise (159.8) |
| 4096 | mamba2 (236.4) < alt (250.1) < headwise (255.3) < attn (288.9) |
| 8192 | mamba2 (221.6) < alt (234.7) < headwise (240.4) < attn (267.8) |

The c4 ordering at seq=4096 carries over to seq=8192 — same
mamba2 → alt → headwise → attn ranking, with similar relative gaps.

## Headline takeaways

1. **Doubling tokens via 2× context only gave a modest ~6–7% c4 / ~6–7% wt
   improvement** for every arch. Mamba-2 did *not* benefit
   differentially from longer context — the relative gap to attn at
   seq=8192 is `attn − mamba2 = +20.9% c4`, virtually identical to
   `+22.2% c4` at seq=4096. Longer context per gradient step is
   roughly arch-neutral at this eval.
2. **seq=8192 with 2× Chinchilla still loses to seq=2048 with 1×
   Chinchilla.** Doubling context cost 1.5–2× the wall-clock time per
   step and bought back only a fraction of the regression introduced
   by halving the optimizer-step count back at seq=2048→4096. Net:
   spending the doubled compute on seq_len is not as good as spending it
   on more gradient steps.
3. **`alt_attn_mamba2` continues to be the second-best arch behind
   pure mamba2 at long context** — and the best on the previous
   seq=2048 sweep on wikitext + piqa. The hybrid travels well across
   seq lengths.
4. **The eval doesn't actually probe long-context capability.** c4_en
   and wikitext_103 use short eval sequences (~1024 tokens). Even if
   the model has better long-range modeling at seq=8192, this
   evaluation can't see it. Showing real long-context wins requires a
   passkey/needle-in-haystack or BookSum-style eval.

## Implication

The hypothesis "longer context per gradient step helps SSMs absolutely"
is **not supported** by these numbers — the modest improvement is
arch-neutral, and the absolute perplexity is still much worse than
the seq=2048 / 1× Chinchilla baseline. To genuinely demonstrate a
long-context advantage, future work should (a) use long-context eval
benchmarks, and/or (b) train at constant Chinchilla *and* constant
step count (which means doubling global batch when doubling seq, not
doubling tokens).

## Run/log/checkpoint locations

- Configs: `configs/lm/scale_50m_seq8192/{attn,mamba2,alt_attn_mamba2,headwise_mamba2}.yaml`
- Logs: `/scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/logs/seq8192/50m-seq8192-{arch}.log`
- Final checkpoints: `/scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-50m-seq8192-{arch}/step477/`
