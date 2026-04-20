"""Aggregate per-run history.json files into a single CSV of final metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs")
    p.add_argument("--out", default="/scratch/gpfs/EHAZAN/tharuntk/mechbench/suite_results.csv")
    args = p.parse_args()

    rows = []
    for run in Path(args.runs_dir).glob("*/history.json"):
        data = json.loads(run.read_text())
        hist = data.get("history", [])
        if not hist:
            continue
        final = hist[-1]
        name = run.parent.name
        # expect task-arch-scale
        parts = name.split("-")
        if len(parts) < 3:
            continue
        task, arch, scale = parts[0], parts[1], parts[-1]
        rows.append({
            "run": name,
            "task": task,
            "arch": arch,
            "scale": scale,
            "n_params": data.get("n_params"),
            "final_tok_acc": final.get("tok_acc"),
            "final_seq_acc": final.get("seq_acc"),
            "final_eval_loss": final.get("eval_loss"),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["scale", "task", "arch"])
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
