# mechanistic-bench roadmap

## Pitch
A mechanistic evaluation suite that lets us iterate on novel sequence
architectures **without** pretraining LMs from scratch for every change.
Suite targets four properties:
1. tasks that show up in real downstream workloads
2. tasks *not* saturated by current SOTA architectures
3. cumulative suite score that *correlates with* downstream LM performance
4. task-level failures that reveal real architectural gaps

## NeurIPS submission plan (deadline in ~2 weeks)
- 1M-param sweep on the full mechanistic suite (all tasks × all archs) — ~1h/arch on 1×H100.
- 10M-param sweep on the full mechanistic suite — ~4–8h/arch.
- 150M language modelling runs per architecture to establish the
  mech-suite ↔ LM correlation — ~5 H100-hrs per model at Chinchilla
  optimality (see docs/COMPUTE.md).

## Architectures
transformer (flash-attn), mamba, rnn, lstm, mlp, alt_attn_mamba (layer-wise hybrid),
headwise (head-wise attn+mamba hybrid). See `src/mechbench/configs/presets.py`.

## Suite tasks
Retrieval: copy, copy_offset, reverse_copy, induction, induction_gap,
multi_induction, associative, short_associative, selective_copy, needle.
Aggregation: counting, parity, cumulative_sum, state_tracking, mode, sort.

## Immediate TODOs (post-scaffold)
- [ ] Install flash-attn (`uv pip install flash-attn --no-build-isolation`) and
      mamba-ssm (`uv pip install mamba-ssm causal-conv1d`) on the H100 node.
- [ ] Re-measure and re-tune `D_MODEL_OFFSETS` so all archs match params to
      within ±10% at each scale.
- [ ] Add a 150M LM trainer (`scripts/train_lm.py`) using a streaming text
      dataset (C4 / The Pile / DCLM) — currently only synthetic tasks.
- [ ] Add an "aggregate suite score" eval that reports normalised per-task
      accuracy + overall ranking.
