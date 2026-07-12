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
- `lm/scale_150m` — ~150M language models for the suite↔LM correlation study.

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

Every task-training run is defined by two YAMLs:

- `configs/scale_<scale>/models/<arch>.yaml` — architecture + training defaults.
- `configs/scale_<scale>/tasks.yaml` — per-task `seq_len`, `vocab_size`, and
  `task_params`, with optional per-task overrides for training fields
  (`lr`, `warmup_steps`, …).

Merge order (lowest → highest): model YAML < `tasks.defaults` < `tasks.<name>`
< CLI flags. Each run dumps its resolved config to `<out_dir>/<run_name>/config.yaml`
so every artifact is self-describing.

```bash
# single run (one model, one task)
uv run python scripts/train.py --scale 1m --model transformer --task induction

# smoke test with CLI overrides
uv run python scripts/train.py --scale 1m --model transformer --task induction \
    --max_steps 200 --batch_size 64

# sweep one model across every task in the scale's tasks.yaml
uv run python scripts/run.py --scale 10m --model transformer

# subset sweep
uv run python scripts/run.py --scale 10m --model mamba --tasks k_hop two_hop
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
uv run python scripts/train_lm.py configs/lm/scale_10m/mechbench.yaml
```

The example YAML trains an 8-layer attention-only 10M-param model on a C4
subset and evaluates against `c4_en` / `wikitext_103` perplexity plus the
`piqa` / `hellaswag` ICL tasks. Checkpoints land under
`/scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs/<run_name>/`.

### Overrides

Any trailing `key=value` arguments are applied as OmegaConf dotlist
overrides on top of the YAML:

```bash
uv run python scripts/train_lm.py configs/lm/scale_10m/mechbench.yaml \
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
uv run python scripts/eval_lm.py configs/lm/scale_10m/mechbench.yaml \
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

## Paper

This suite is the screening battery from:

> **Mechanistic Capability Probes as a Cheap Screen for Sequence-Mixer
> Architectures.** Kia Ghods, Tharun Kumar Tiruppali Kalidoss, Elad Hazan.
> ICML 2026 Workshop on Mechanistic Interpretability.

The analysis scripts that produce the paper's correlation figures live in
`analysis/` (unified 1M accuracy matrix, 50M/150M LM cross-entropy tables,
drop-primitive ablation, rank-correlation figures), and the 5-seed
replication of the 1M sweep is in `figures/seed_sweep/`.

```bibtex
@inproceedings{ghods2026mechprobes,
  title     = {Mechanistic Capability Probes as a Cheap Screen for Sequence-Mixer Architectures},
  author    = {Ghods, Kia and Tiruppali Kalidoss, Tharun Kumar and Hazan, Elad},
  booktitle = {ICML Workshop on Mechanistic Interpretability},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
