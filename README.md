# mechanistic-bench

A mechanistic evaluation suite for sequence modeling architectures. The goal is
to enable fast iteration on novel architectures by evaluating on targeted
synthetic tasks **before** spending compute to train large downstream models.

## Design criteria

The suite is built around four requirements for each task:

1. **Useful** — the capability shows up in downstream workloads.
2. **Unsaturated** — modern architectures still fail at it.
3. **Predictive** — cumulative performance on the suite correlates with
   downstream task performance.
4. **Diagnostic** — failure reveals real gaps in current architectures.

## Scales

- `scale_1m`   — ~1M params, sanity check & rapid sweeps.
- `scale_10m`  — ~10M params, headline mechanistic-suite numbers.
- `lm_150m`    — ~150M language models for the suite↔LM correlation study.

## Architectures

Transformer (flash-attn), RNN, LSTM, MLP-mixer, Mamba, and two hybrids —
alternating attention+mamba **layer-wise** and **head-wise**.

## Quickstart

```bash
uv sync
# optional (Linux + CUDA only):
uv pip install flash-attn --no-build-isolation
uv pip install mamba-ssm causal-conv1d

# tiny smoke test
uv run python scripts/train.py \
    --task induction --arch transformer --scale 1m \
    --max_steps 200 --batch_size 64 --seq_len 128

# full suite dispatch
uv run python scripts/run_suite.py --scale 1m
```

Checkpoints and wandb artifacts are written under
`/scratch/gpfs/EHAZAN/tharuntk/mechbench/`.
