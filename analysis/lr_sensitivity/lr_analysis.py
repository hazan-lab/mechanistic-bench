#!/usr/bin/env python3
"""LR-sensitivity analysis: does the arch ranking survive LR changes?
LRs: 1e-4, 3e-4 (default, from seed sweep seed1), 1e-3.
Archs: attn, mamba, stu, alt_attn_mamba. 11 headline tasks."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path

S = Path(__file__).parent
lr = pd.read_csv(S / "lr_sweep.csv")
# normalize lr column to canonical strings (csv stored 0.0001 / 0.001)
lr["lr"] = lr["lr"].map(lambda v: {1e-4: "1e-4", 1e-3: "1e-3"}.get(float(v), str(v)))
seed = pd.read_csv(S / "cluster/seed_sweep/seed_sweep_tokacc.csv")

ARCHS = ["attn", "mamba", "stu", "alt_attn_mamba"]
TASKS = ["copy", "induction", "associative", "selective_copy", "needle",
         "counting", "parity", "state_tracking", "copy_count", "state_retrieve",
         "selective_parity"]

# default LR = 3e-4 from seed1 column of the 5-seed sweep
d = seed[seed.task.isin(TASKS) & seed.arch.isin(ARCHS)][["task", "arch", "seed1"]]
d = d.rename(columns={"seed1": "tok_acc"}); d["lr"] = "3e-4"
allr = pd.concat([lr[["lr", "task", "arch", "tok_acc"]], d], ignore_index=True)
# check coverage
print("rows per lr:", allr.groupby("lr").size().to_dict())
miss = [(t, a) for t in TASKS for a in ARCHS
        if allr[(allr.task == t) & (allr.arch == a)].lr.nunique() < 3]
print("incomplete (task,arch) across 3 LRs:", miss)

LRS = ["1e-4", "3e-4", "1e-3"]
# suite mean per (lr, arch)
pivot = allr[allr.task.isin(TASKS)].groupby(["lr", "arch"]).tok_acc.mean().unstack()
pivot = pivot.reindex(index=LRS, columns=ARCHS)
print("\n=== suite mean (11 tasks) per LR x arch ===")
print((pivot * 100).round(1))

print("\n=== arch ranking per LR (best->worst) ===")
for l in LRS:
    order = pivot.loc[l].sort_values(ascending=False)
    print(f"  {l}: " + " > ".join(f"{a}({v*100:.0f})" for a, v in order.items()))

print("\n=== cross-LR Spearman of the 4-arch suite-mean ranking ===")
for i in range(len(LRS)):
    for j in range(i + 1, len(LRS)):
        rho, _ = spearmanr(pivot.loc[LRS[i]], pivot.loc[LRS[j]])
        print(f"  {LRS[i]} vs {LRS[j]}: rho={rho:+.3f}")

# best-LR-per-arch: does any arch's optimum flip the ranking?
print("\n=== per-arch best LR (does tuning change the order?) ===")
best_lr = pivot.idxmax()
print("best LR per arch:", best_lr.to_dict())
oracle = pd.Series({a: pivot[a].max() for a in ARCHS})
print("oracle (best-LR) suite means:", (oracle * 100).round(1).to_dict())
print("oracle ranking:", " > ".join(oracle.sort_values(ascending=False).index))
rho, _ = spearmanr(oracle, pivot.loc["3e-4"])
print(f"oracle vs default-LR ranking rho={rho:+.3f}")

# save tidy table for the paper
out = (pivot * 100).round(1)
out.to_csv(S / "lr_suite_means.csv")
print("\nwrote lr_suite_means.csv")
