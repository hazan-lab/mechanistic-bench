# Compute budget (rough)

Based on Chinchilla-optimal 20 tok/param and H100 bf16 at ~40% MFU:

- **1M params, full suite (16 tasks × 7 archs × 4000 steps, seq 256, bs 128)**
  ~112 runs × ~10 min/run = **~20 H100-hours** total.

- **10M params, full suite** — ~112 runs × ~30–60 min/run = **~60–100 H100-hours**.

- **150M params, language modelling (3B tokens)**
  ~**5 H100-hours** per architecture on 1×H100 (or ~2h on 4×H100).
  For 7 architectures, **~35 H100-hours** total (plus headroom for dev).

**Total ballpark for NeurIPS submission: 150–250 H100-hours.** At $2–3/H100-hr
this is $300–750 of cloud compute, or 2–3 days of an 8-GPU node.

Optimisations in scope:
- FlashAttention-2 for transformer/headwise (~20–30% speedup end-to-end).
- FlashFFTConv not required (Mamba uses selective-scan, not FFT conv).
- torch.compile: +10–25% once graph breaks are audited.
