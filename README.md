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

## Language modeling

In addition to the synthetic mechanistic tasks, the repo ships a single-GPU
language-modeling trainer ported from the OLMo / spectral-transformers
pipeline. This lets you pretrain a `MechModel` on real text (preprocessed
memmaps) and score it against held-out perplexity corpora and a handful of
in-context-learning downstream tasks.

### Running

```bash
uv run python scripts/train_lm.py configs/lm/mechbench-10m.yaml
```

The example YAML trains an 8-layer attention-only 10M-param model on a C4
subset and evaluates against `c4_en` / `wikitext_103` perplexity plus the
`piqa` / `hellaswag` ICL tasks. Checkpoints land under
`/scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/<run_name>/`.

### Overrides

Any trailing `key=value` arguments are applied as OmegaConf dotlist
overrides on top of the YAML:

```bash
uv run python scripts/train_lm.py configs/lm/mechbench-10m.yaml \
    optimizer.learning_rate=1e-4 \
    max_duration=500 \
    model.block_types='[mamba,mamba,mamba,mamba,mamba,mamba,mamba,mamba]'
```

Pass `--no-validate-paths` to skip existence checks on data paths (useful
for dry configs on machines without the full dataset mount).

### Standalone evaluation

`scripts/eval_lm.py` runs just the evaluators in a YAML against a saved
checkpoint, without training:

```bash
uv run python scripts/eval_lm.py configs/lm/mechbench-10m.yaml \
    /scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/mechbench-10m-attn/step1000/model.pt
```

### Fixture symlinks

The LM trainer expects three data directories under
`src/mechbench/olmo_data/`:

- `hf_datasets/`    — cached HuggingFace ICL datasets
- `oe_eval_tasks/`  — `requests.jsonl(.gz)` files for ICL eval
- `tokenizers/`     — tokenizer JSONs (GPT-NeoX / Dolma2 / …)

These are checked into the sibling `spectral-transformers` checkout and
should be symlinked in. If the links are missing, recreate them with:

```bash
ln -s /home/tt6444/spectral-transformers/olmo_data/hf_datasets \
      src/mechbench/olmo_data/hf_datasets
ln -s /home/tt6444/spectral-transformers/olmo_data/oe_eval_tasks \
      src/mechbench/olmo_data/oe_eval_tasks
ln -s /home/tt6444/spectral-transformers/olmo_data/tokenizers \
      src/mechbench/olmo_data/tokenizers
```

The preprocessed training / perplexity-eval memmaps live outside the repo
under `/scratch/gpfs/EHAZAN/tharuntk/OLMo-data/{preprocessed,eval-data}/`.

### Known limitations

- Single-GPU only — FSDP / DDP were deliberately not ported. Hooks for
  `global_train_batch_size` vs. `device_train_microbatch_size` exist, but
  they currently drive gradient accumulation on one device.
- Downstream tasks requiring `pmi_dc` (e.g. `boolq`, `arc_challenge`) are
  not wired up — those require an extra domain-conditional forward pass
  that the trainer does not currently issue.
- `torch.compile`, activation checkpointing, and sharded checkpointers are
  not ported.
- Token-valued `max_duration` (e.g. `"100000T"`) is rejected; configure
  `max_duration` as a plain step count.
