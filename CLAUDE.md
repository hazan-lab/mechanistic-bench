# mechanistic-bench — Claude notes

## Environment pins

This project targets **Python 3.12 / torch 2.9.1+cu128 / cp312-linux_x86_64**.
The pin exists so `flash-attn`, `mamba-ssm`, and `causal-conv1d` can install
from prebuilt wheels without a source compile.

Relevant `pyproject.toml` bits:

- `torch>=2.9,<2.10` in `[project]`
- `[[tool.uv.index]]` `pytorch-cu128` → `https://download.pytorch.org/whl/cu128`
- `[tool.uv.sources]` maps `torch` to that index and pins
  `flash-attn` / `mamba-ssm` / `causal-conv1d` to specific GitHub release wheels
  (`cu12torch2.9cxx11abiTRUE-cp312`).

**Do not bump torch past 2.9.x** without first checking that matching wheels
exist for all three GPU libs — at the time of writing, mamba-ssm's newest
wheel is `cu12torch2.10` but flash-attn 2.8.3 only publishes up to
`cu12torch2.9`, so 2.9 is the highest version where all three overlap.

## `uv sync` quirk

Switching torch families (cu13 → cu12) leaves stale CUDA sub-libraries on
disk even though uv's metadata shows them installed. Symptom: `import torch`
fails with `libcudnn.so.9` / `libcusparseLt.so.0` / `libnccl.so.*` not found.

Fix: `uv pip install --force-reinstall <missing-package>` (e.g.
`nvidia-cudnn-cu12`, `nvidia-cusparselt-cu12`, `nvidia-nccl-cu12`,
`nvidia-nvshmem-cu12`).

## Build-from-source fallback

If wheels ever stop matching, source compiles need the CUDA toolkit on PATH.
On Della: `module load cudatoolkit/13.0` (or the version matching
`torch.version.cuda`) provides `nvcc`. Expect 30+ minutes per lib and high
memory; cap with `MAX_JOBS=4`.

## Storage

Per the user's global rules, all datasets, checkpoints, and large artifacts
live under `/scratch/gpfs/EHAZAN/tharuntk/` — never in the project or home
directory.
