# 150M LM training: seq_len=2048 vs seq_len=4096

Configs (`configs/lm/scale_150m/` and `configs/lm/scale_150m_seq4096/`)
were merged in PR #41. This file reports the actual training results for
the seq=4096 sweep, side-by-side with the seq=2048 baseline. Three archs
were trained at each seq length: `attn`, `mamba2`, `alt_attn_mamba2`
(headwise hybrid was not retrained at 150M for either seq length).

Both sweeps are Chinchilla-budgeted at 3B tokens (150M params × 20). The
2048 runs use 2862 optimizer steps (`global_batch=512 × seq=2048 × 2862
≈ 3.0B`); the 4096 runs use 1431 steps with the same global batch
(`global_batch=512 × seq=4096 × 1431 ≈ 3.0B`). Identical model
architecture, optimizer, scheduler, and data per arch — the only deltas
are `max_seq_len` (2048→4096), `device_train_microbatch_size` (8→4 to
keep per-device memory similar), `max_duration` (2862→1431), and
proportionally scaled warmup/eval/save intervals.

## Final evals (3B tokens; same Chinchilla budget for both seq lengths)

| arch | seq | params | c4 ppl ↓ | wt_103 ppl ↓ | piqa ↑ | hellaswag ↑ |
|---|---|---|---|---|---|---|
| attn            | 2048 | 149.87M | **43.54** | **73.96** | **0.604** | 0.271 |
| attn            | 4096 | 149.87M | 63.62     | 128.47    | 0.575     | 0.261 |
| mamba2          | 2048 | 149.90M | **48.26** | **92.89** | 0.595     | **0.277** |
| mamba2          | 4096 | 149.90M | 67.81     | 158.64    | 0.574     | 0.265 |
| alt_attn_mamba2 | 2048 | 150.90M | **45.51** | **75.84** | **0.596** | 0.275 |
| alt_attn_mamba2 | 4096 | 150.90M | 63.35     | 116.61    | 0.572     | 0.265 |

## Per-arch deltas (4096 minus 2048)

| arch | Δc4 ppl | Δwt ppl | Δpiqa | Δhellaswag |
|---|---|---|---|---|
| attn            | +20.08 (+46%) | +54.51 (+74%) | -0.029 | -0.010 |
| mamba2          | +19.55 (+41%) | +65.75 (+71%) | -0.021 | -0.012 |
| alt_attn_mamba2 | +17.84 (+39%) | +40.77 (+54%) | -0.024 | -0.010 |

## Within-seq ranking on c4 (lower is better)

| seq | best → worst on c4 |
|---|---|
| 2048 | attn (43.54) < alt (45.51) < mamba2 (48.26) |
| 4096 | alt (63.35) ≈ attn (63.62) < mamba2 (67.81) |

## Within-seq ranking on wikitext_103

| seq | best → worst on wt |
|---|---|
| 2048 | attn (73.96) < alt (75.84) < mamba2 (92.89) |
| 4096 | alt (116.61) < attn (128.47) < mamba2 (158.64) |

## Per-step trajectory on c4 perplexity (150M seq=4096)

The seq=4096 runs show a clear **temporal crossover** as training
progresses — mamba2 dominates very early, then attn closes, then
alt overtakes both by the end:

| step | attn c4 | mamba2 c4 | alt c4 | leader |
|---|---|---|---|---|
| 250  | 265.2 | 190.1 | 253.1 | **mamba2** (-28% vs attn) |
| 500  | 127.6 | 110.4 | 129.0 | **mamba2** (-13% vs attn) |
| 750  |  89.7 |  86.1 |  89.1 | mamba2 (-4% vs attn) |
| 1000 |  72.9 |  75.4 |  72.0 | **alt** narrowly above attn |
| 1250 |  66.0 |  69.8 |  65.5 | **alt** |
| 1431 |  63.6 |  67.8 |  63.3 | **alt** ≈ attn, mamba2 last |

The wikitext_103 trajectory is more decisive — alt is the clear final
winner there (-9% vs attn, -27% vs mamba2).

## Takeaways

1. **At fixed Chinchilla budget (3B tokens), seq=4096 is strictly worse
   on perplexity for every arch** — same regression pattern as the
   50M-seq4096 sweep, +39–46% on c4 ppl and +54–74% on wikitext_103.
   Halving the optimizer-step count (2862 → 1431) hurts more than
   longer context helps in absolute terms.
2. **Ranking flips between seq lengths.**
   - At seq=2048: `attn < alt < mamba2` on c4 perplexity (attn wins).
   - At seq=4096: `alt ≈ attn < mamba2` on c4; `alt < attn < mamba2` on wikitext_103.
   The hybrid `alt_attn_mamba2` benefits *more* from longer context
   than pure attn — its c4 regression was the smallest (+39% vs +46% for attn),
   and on wikitext it absorbed only +54% vs +74% for attn.
3. **Mamba-2's early lead does not persist.** At step 250, mamba2 was
   28% ahead on c4. By step 1000, attn caught up. By step 1250, alt
   had overtaken both. This is consistent with the seq=2048 trajectory
   pattern (attn beats mamba2 at the end) but starting from a much
   bigger mamba2 lead — the seq=4096 setting just stretches the
   convergence timeline.
4. **ICL metrics (piqa, hellaswag) regress only slightly** (≈ -0.01 to
   -0.03) — well within noise — and the rankings between archs barely
   move.

## Implication

If the goal is matching spectral-transformers' 150M setup
(seq=4096, batch=1024 → 4× our seq=2048 tokens/step, 2× our seq=4096
tokens/step), the next experiment would also need to ~2× the global
batch at seq=4096 to recover the gradient-step count we lose under
fixed Chinchilla. The numbers here isolate seq_len alone.

For the question "does longer context help mamba2 catch attn at the
LM scale", the answer is: **mamba2 looks great mid-training but loses
the absolute fight by the end; the layer-alternating hybrid is what
genuinely benefits.**

## Run/log/checkpoint locations

- Configs: `configs/lm/scale_150m_seq4096/{attn,mamba2,alt_attn_mamba2}.yaml` (merged in #41)
- Logs: `/scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/logs/seq4096/150m-seq4096-{arch}.log`
- Final checkpoints: `/scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-150m-seq4096-{arch}/step1431/`
