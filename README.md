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

## Installation

The project pins **torch 2.9.1 + CUDA 12.8** so that prebuilt wheels for
`flash-attn`, `mamba-ssm`, and `causal-conv1d` are available (no source
compile). The three libs are declared in `[tool.uv.sources]` in
`pyproject.toml` with direct GitHub release URLs.

```bash
# core only
uv sync

# with GPU kernels (Linux + CUDA, cp312)
uv sync --extra flash --extra mamba
```

If `uv sync` ever leaves stale CUDA sub-libraries on disk (symptom:
`ImportError: libcudnn.so.9` / `libcusparseLt.so.0` / similar at `import
torch`), force-reinstall the missing one:

```bash
uv pip install --force-reinstall nvidia-cudnn-cu12
# or: nvidia-cusparselt-cu12, nvidia-nccl-cu12, nvidia-nvshmem-cu12
```

## Quickstart

```bash
# tiny smoke test
uv run python scripts/train.py \
    --task induction --arch transformer --scale 1m \
    --max_steps 200 --batch_size 64 --seq_len 128

# full suite dispatch
uv run python scripts/run_suite.py --scale 1m
```

Checkpoints and wandb artifacts are written under
`/scratch/gpfs/EHAZAN/tharuntk/mechbench/`.
