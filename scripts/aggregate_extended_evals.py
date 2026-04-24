"""Aggregate per-run JSONs from eval_lm_extended into a markdown table.

Input: results/raw/{run-name}.json files
Output: results/extended_evals.md  (per-checkpoint table, summary, winners)
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

RAW_DIR = Path("results/raw")
OUT_MD = Path("results/extended_evals.md")
OUT_CSV = Path("results/extended_evals.csv")

# Display order for the run-names (columns in the summary table).
RUN_ORDER = [
    "150m-attn",
    "150m-mamba2",
    "150m-alt-attn-mamba2",
    "50m-attn",
    "50m-mamba2",
    "50m-alt-attn-mamba2",
    "50m-headwise-mamba2",
]

# Higher-is-better unless the metric name mentions loss/ppl/bpb/ce.
LOWER_IS_BETTER_HINTS = ("ce_loss", "_loss", "_ppl", "_bpb", "bits_per_byte")


def is_higher_better(metric_name: str) -> bool:
    n = metric_name.lower()
    return not any(h in n for h in LOWER_IS_BETTER_HINTS)


def load_raw() -> "OrderedDict[str, dict]":
    out: "OrderedDict[str, dict]" = OrderedDict()
    for name in RUN_ORDER:
        p = RAW_DIR / f"{name}.json"
        if not p.exists():
            print(f"WARN: missing {p}", file=sys.stderr)
            continue
        with open(p) as f:
            out[name] = json.load(f)
    for p in sorted(RAW_DIR.glob("*.json")):
        n = p.stem
        if n not in out:
            with open(p) as f:
                out[n] = json.load(f)
    return out


def collect_rows(raw: "OrderedDict[str, dict]") -> dict:
    """Returns {task_label: {run_name: {metric_key: value}}}."""
    all_rows: dict = {}
    for run, blob in raw.items():
        for task, res in blob["results"].items():
            all_rows.setdefault(task, {})
            if res.get("ok"):
                all_rows[task][run] = res["metrics"]
            else:
                all_rows[task][run] = {"__error__": res.get("error", "?")}
    return all_rows


def write_csv(raw, rows):
    runs = list(raw.keys())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write("task,metric," + ",".join(runs) + "\n")
        for task in sorted(rows.keys()):
            metric_keys: set = set()
            for run in runs:
                d = rows[task].get(run, {})
                if "__error__" not in d:
                    metric_keys.update(d.keys())
            for mk in sorted(metric_keys):
                vals = []
                for run in runs:
                    d = rows[task].get(run, {})
                    if "__error__" in d:
                        vals.append("ERR")
                    elif mk in d:
                        vals.append(f"{d[mk]:.4f}")
                    else:
                        vals.append("")
                f.write(f"{task},{mk}," + ",".join(vals) + "\n")
    print(f"wrote {OUT_CSV}")


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def pick_primary_metric(metric_keys: list[str]) -> str | None:
    """Pick the 'headline' metric for a task when multiple keys exist."""
    preferred_tails = ("_len_norm", "_acc", "_f1", "_ce_loss", "_bpb")
    for tail in preferred_tails:
        for mk in metric_keys:
            if mk.endswith(tail):
                return mk
    return metric_keys[0] if metric_keys else None


def format_run_scores(rows, task, runs):
    """Return list of (run, primary_metric, value_or_None, is_error)."""
    # Collect metric keys across runs for this task
    all_mks: list[str] = []
    for run in runs:
        d = rows[task].get(run, {})
        if "__error__" not in d:
            for k in d.keys():
                if k not in all_mks:
                    all_mks.append(k)
    primary = pick_primary_metric(all_mks)
    scores = []
    for run in runs:
        d = rows[task].get(run, {})
        if "__error__" in d:
            scores.append((run, primary, None, True))
        elif primary and primary in d:
            scores.append((run, primary, d[primary], False))
        else:
            scores.append((run, primary, None, False))
    return primary, scores


def main():
    raw = load_raw()
    rows = collect_rows(raw)
    runs = list(raw.keys())

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("# Extended LM downstream evals\n\n")
        f.write("Checkpoints: `mechbench-{150m,50m}-{attn,mamba2,alt-attn-mamba2,headwise-mamba2}` "
                "at their final step (2862 for 150m, 954 for 50m).\n\n")
        f.write("Evaluator script: `scripts/eval_lm_extended.py`. Raw per-run JSONs under "
                "`results/raw/`.\n\n")
        f.write(f"Runs included: {', '.join(runs)}\n\n")

        # --- Summary table: headline metric per task ---
        f.write("## Summary (headline metric per task)\n\n")
        summary_header = ["task", "metric"] + runs + ["winner"]
        summary_rows: list[list[str]] = []
        for task in sorted(rows.keys()):
            primary, scores = format_run_scores(rows, task, runs)
            if primary is None:
                continue
            higher = is_higher_better(primary)
            vals = [s[2] for s in scores if s[2] is not None]
            if not vals:
                winner_name = "-"
                val_strs = ["ERR" if s[3] else "-" for s in scores]
            else:
                best = max(vals) if higher else min(vals)
                winner_name = ", ".join([s[0] for s in scores if s[2] == best]) or "-"
                val_strs = []
                for s in scores:
                    if s[3]:
                        val_strs.append("ERR")
                    elif s[2] is None:
                        val_strs.append("-")
                    elif s[2] == best:
                        val_strs.append(f"**{s[2]:.4f}**")
                    else:
                        val_strs.append(f"{s[2]:.4f}")
            summary_rows.append([task, primary] + val_strs + [winner_name])
        f.write(md_table(summary_header, summary_rows))
        f.write("\n")

        # --- Per-scale winner count ---
        f.write("## Per-scale win counts (headline metric)\n\n")
        scales = {
            "150m": [r for r in runs if r.startswith("150m")],
            "50m": [r for r in runs if r.startswith("50m")],
        }
        win_rows = []
        for scale, scale_runs in scales.items():
            counts = {r: 0 for r in scale_runs}
            total = 0
            for task in rows:
                primary, scores = format_run_scores(rows, task, scale_runs)
                if primary is None:
                    continue
                higher = is_higher_better(primary)
                vals = [s[2] for s in scores if s[2] is not None]
                if not vals:
                    continue
                best = max(vals) if higher else min(vals)
                for s in scores:
                    if s[2] == best:
                        counts[s[0]] += 1
                total += 1
            win_rows.append([scale, str(total)] + [str(counts[r]) for r in scale_runs])
        # Unify columns
        for scale, scale_runs in scales.items():
            f.write(f"### {scale}\n\n")
            counts = {r: 0 for r in scale_runs}
            total = 0
            for task in rows:
                primary, scores = format_run_scores(rows, task, scale_runs)
                if primary is None:
                    continue
                higher = is_higher_better(primary)
                vals = [s[2] for s in scores if s[2] is not None]
                if not vals:
                    continue
                best = max(vals) if higher else min(vals)
                for s in scores:
                    if s[2] == best:
                        counts[s[0]] += 1
                total += 1
            header = ["arch"] + ["wins"]
            body = [[r, str(counts[r])] for r in scale_runs]
            body.append(["total tasks scored", str(total)])
            f.write(md_table(header, body))
            f.write("\n")

        # --- Per-task detail (all metric keys) ---
        f.write("## Per-task detail (all reported metric keys)\n\n")
        for task in sorted(rows.keys()):
            f.write(f"### {task}\n\n")
            metric_keys: list[str] = []
            for run in runs:
                d = rows[task].get(run, {})
                if "__error__" not in d:
                    for k in d.keys():
                        if k not in metric_keys:
                            metric_keys.append(k)
            if not metric_keys:
                f.write("All runs errored.\n\n")
                for run in runs:
                    d = rows[task].get(run, {})
                    if "__error__" in d:
                        f.write(f"- `{run}`: `{d['__error__']}`\n")
                f.write("\n")
                continue
            header = ["run"] + metric_keys
            body = []
            for run in runs:
                d = rows[task].get(run, {})
                if "__error__" in d:
                    body.append([run] + [f"ERR: {d['__error__'][:60]}"] + [""] * (len(metric_keys) - 1))
                else:
                    body.append([run] + [f"{d[mk]:.4f}" if mk in d else "-" for mk in metric_keys])
            f.write(md_table(header, body))
            f.write("\n")

    print(f"wrote {OUT_MD}")
    write_csv(raw, rows)


if __name__ == "__main__":
    main()
