"""Aggregate 1m mech-suite per-task accuracy and 50M/150M LM CE into one table.

Outputs:
    workshop_analysis/unified_1m.csv   (rows=tasks, cols=architectures)
    workshop_analysis/lm_ce.csv        (rows=architectures, cols=scale/eval CE)
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROJECT_FIGURES = Path("/home/tt6444/mechanistic-bench/figures")
ANALYSIS = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/analysis/20260423")
LM_RUNS = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench-lm/runs")
OUT = ROOT / "workshop_analysis"

# 1) per-task @ 1m -- union of three CSVs covering 9 architectures
csvs = [
    PROJECT_FIGURES / "alt_attn_stu_1m.csv",
    PROJECT_FIGURES / "headwise_stu_1m.csv",
    PROJECT_FIGURES / "mamba2_hybrids_1m.csv",
]

frames = []
for c in csvs:
    df = pd.read_csv(c).set_index("task")
    frames.append(df)

# Union of all task indices and arch columns
all_tasks = sorted(set().union(*(f.index for f in frames)))
all_archs = []
for f in frames:
    for col in f.columns:
        if col not in all_archs:
            all_archs.append(col)

wide = pd.DataFrame(index=all_tasks, columns=all_archs, dtype=float)
for f in frames:
    for col in f.columns:
        # Fill in values from this frame where they exist
        common = wide.index.intersection(f.index)
        # Only fill where wide is NaN (later frames don't overwrite)
        existing = wide[col].notna()
        update = (~existing) & wide.index.isin(common)
        wide.loc[update, col] = f.loc[wide.index[update], col].values

# 2) Add lstm + rnn + mlp from final_metrics.csv where they exist
fm = pd.read_csv(ANALYSIS / "final_metrics.csv")
m1 = fm[fm.scale == "1m"].pivot_table(index="task", columns="arch", values="tok_acc", aggfunc="mean")
extra_archs = ["lstm", "mamba2", "mamba", "stu", "attn", "alt_attn_mamba", "headwise"]
for col in m1.columns:
    if col not in wide.columns:
        wide[col] = pd.NA
    common = wide.index.intersection(m1.index)
    update = wide[col].isna() & wide.index.isin(common)
    wide.loc[update, col] = m1.loc[wide.index[update], col].values

wide.index.name = "task"
wide = wide.sort_index()
wide.to_csv(OUT / "unified_1m.csv")
print(f"wrote {OUT/'unified_1m.csv'} shape={wide.shape}")
print("archs:", list(wide.columns))
print("ntasks:", len(wide.index))
nan_arch = wide.isna().sum(axis=0)
print("\nNaN-task counts per arch:")
print(nan_arch.to_string())

# 3) LM CE per arch at 50M and 150M
def collect_lm(scale_tag: str) -> pd.DataFrame:
    rows = []
    for d in sorted(LM_RUNS.glob(f"mechbench-{scale_tag}-*")):
        if "seq" in d.name:
            continue
        try:
            h = json.load(open(d / "history.json"))
            f = h[-1]
            rows.append({
                "scale": scale_tag,
                "arch": d.name.replace(f"mechbench-{scale_tag}-", "").replace("-", "_"),
                "step": f.get("step"),
                "c4_en_CE": f["eval/eval/c4_en/CrossEntropyLoss"],
                "wikitext_103_CE": f["eval/eval/wikitext_103/CrossEntropyLoss"],
                "hellaswag": f["eval/eval/downstream/hellaswag_len_norm"],
                "piqa": f["eval/eval/downstream/piqa_len_norm"],
            })
        except FileNotFoundError:
            continue
    return pd.DataFrame(rows)

lm = pd.concat([collect_lm("50m"), collect_lm("150m")], ignore_index=True)
lm.to_csv(OUT / "lm_ce.csv", index=False)
print("\nLM CE:")
print(lm.to_string(index=False))
