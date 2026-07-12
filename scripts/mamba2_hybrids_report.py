"""Build the mamba2-hybrids vs baselines comparison report.

Writes:
- /home/tt6444/mechanistic-bench/figures/mamba2_hybrids_1m.csv
- /home/tt6444/mechanistic-bench/figures/mamba2_hybrids_10m.csv
- /home/tt6444/mechanistic-bench/figures/mamba2_hybrids_report.md
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HYBRIDS_DIR = Path("/scratch/gpfs/EHAZAN/tharuntk/mech_runs/mamba2_hybrids")
BASELINES_OLD = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs")
BASELINES_NEW = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs_20260420_212136")
SWEEP_10M = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/sweep_10m_hard")
SWEEP_10M_LSTM = SWEEP_10M / "lstm"
SWEEP_10M_STU = SWEEP_10M / "stu"

OUT_DIR = Path("/home/tt6444/mechanistic-bench/figures")
OUT_1M = OUT_DIR / "mamba2_hybrids_1m.csv"
OUT_10M = OUT_DIR / "mamba2_hybrids_10m.csv"
OUT_MD = OUT_DIR / "mamba2_hybrids_report.md"

# the set of arch tokens we recognise, longest-first so multi-underscore
# archs like alt_attn_mamba2 are matched before mamba2
KNOWN_ARCHS = [
    "alt_attn_mamba2",
    "alt_attn_mamba",
    "headwise_mamba2",
    "headwise",
    "mamba2",
    "mamba",
    "attn",
    "lstm",
    "stu",
]


def parse_run_name(name: str) -> tuple[str, str, str] | None:
    # name = {task}-{arch}-{scale}[-hard]
    rest = name
    if rest.endswith("-hard"):
        rest = rest[: -len("-hard")]
    # find scale
    if rest.endswith("-1m"):
        scale = "1m"
        rest = rest[: -len("-1m")]
    elif rest.endswith("-10m"):
        scale = "10m"
        rest = rest[: -len("-10m")]
    else:
        return None
    # match arch suffix
    for arch in KNOWN_ARCHS:
        suf = "-" + arch
        if rest.endswith(suf):
            task = rest[: -len(suf)]
            return task, arch, scale
    return None


def read_final(run_dir: Path) -> dict | None:
    hj = run_dir / "history.json"
    if not hj.exists():
        return None
    try:
        data = json.loads(hj.read_text())
    except Exception:
        return None
    hist = data.get("history", [])
    if not hist:
        return None
    final = hist[-1]
    tok = final.get("tok_acc")
    loss = final.get("eval_loss")
    nan_tok = tok is None or (isinstance(tok, float) and math.isnan(tok))
    nan_loss = loss is None or (isinstance(loss, float) and math.isnan(loss))
    return {
        "mtime": hj.stat().st_mtime,
        "n_params": data.get("n_params"),
        "final_tok_acc": tok,
        "final_seq_acc": final.get("seq_acc"),
        "final_eval_loss": loss,
        "nan": nan_tok or nan_loss,
        "steps": final.get("step"),
    }


def collect(dirs: list[Path]) -> dict[tuple[str, str, str], dict]:
    """Collect runs keyed by (task, arch, scale). Later entries override if newer mtime."""
    out: dict[tuple[str, str, str], dict] = {}
    for base in dirs:
        if not base.exists():
            continue
        for sub in base.iterdir():
            if not sub.is_dir():
                continue
            # recursive one level if subdir is named lstm/stu (for 10m sweep)
            if (sub / "history.json").exists():
                parsed = parse_run_name(sub.name)
                if parsed is None:
                    continue
                rec = read_final(sub)
                if rec is None:
                    continue
                rec["run_dir"] = str(sub)
                key = parsed
                if key not in out or rec["mtime"] > out[key]["mtime"]:
                    out[key] = rec
            else:
                # nested sweep subdirs (e.g. sweep_10m_hard/lstm/*)
                for inner in sub.iterdir():
                    if not inner.is_dir():
                        continue
                    if not (inner / "history.json").exists():
                        continue
                    parsed = parse_run_name(inner.name)
                    if parsed is None:
                        continue
                    rec = read_final(inner)
                    if rec is None:
                        continue
                    rec["run_dir"] = str(inner)
                    key = parsed
                    if key not in out or rec["mtime"] > out[key]["mtime"]:
                        out[key] = rec
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hybrids = collect([HYBRIDS_DIR])
    baselines = collect([BASELINES_OLD, BASELINES_NEW])
    sweep10 = collect([SWEEP_10M])

    # ------------------------------------------------------------------ 1m
    archs_1m = [
        "alt_attn_mamba",
        "alt_attn_mamba2",
        "headwise",
        "headwise_mamba2",
        "attn",
        "mamba",
        "mamba2",
    ]
    all_tasks_1m = sorted({
        t for (t, a, s) in list(hybrids) + list(baselines) if s == "1m" and a in archs_1m
    })

    def lookup(store, task, arch, scale):
        rec = store.get((task, arch, scale))
        return rec

    rows_1m = []
    for task in all_tasks_1m:
        row = {"task": task}
        for arch in archs_1m:
            rec = lookup(hybrids, task, arch, "1m") or lookup(baselines, task, arch, "1m")
            row[arch] = rec["final_tok_acc"] if rec and not rec["nan"] else None
            row[f"{arch}__n_params"] = rec["n_params"] if rec else None
            row[f"{arch}__nan"] = rec["nan"] if rec else False
        rows_1m.append(row)

    # CSV 1m
    with OUT_1M.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["task"] + archs_1m
        w.writerow(header)
        for r in rows_1m:
            w.writerow([r["task"]] + [r[a] for a in archs_1m])

    # ------------------------------------------------------------------ 10m
    # 10m hybrids from hybrids dir; baselines (attn/lstm/stu) from sweep10
    archs_10m = ["attn", "alt_attn_mamba2", "headwise_mamba2", "lstm", "stu"]
    tasks_10m = sorted({
        t for (t, a, s) in list(hybrids) + list(sweep10) if s == "10m" and a in archs_10m
    })

    rows_10m = []
    for task in tasks_10m:
        row = {"task": task}
        for arch in archs_10m:
            rec = lookup(hybrids, task, arch, "10m") or lookup(sweep10, task, arch, "10m")
            row[arch] = rec["final_tok_acc"] if rec and not rec["nan"] else None
            row[f"{arch}__n_params"] = rec["n_params"] if rec else None
            row[f"{arch}__nan"] = rec["nan"] if rec else False
        rows_10m.append(row)

    with OUT_10M.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["task"] + archs_10m
        w.writerow(header)
        for r in rows_10m:
            w.writerow([r["task"]] + [r[a] for a in archs_10m])

    # ------------------------------------------------------------------ Markdown
    def fmt(v):
        return "—" if v is None else f"{v:.3f}"

    def fmt_delta(v):
        if v is None:
            return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.3f}"

    def highlight(delta, v_str):
        if delta is None:
            return v_str
        if abs(delta) >= 0.1:
            return f"**{v_str}**"
        return v_str

    # 1m head-to-head
    h2h = []
    for r in rows_1m:
        alt1 = r["alt_attn_mamba"]
        alt2 = r["alt_attn_mamba2"]
        hw1 = r["headwise"]
        hw2 = r["headwise_mamba2"]
        d_alt = (alt2 - alt1) if (alt1 is not None and alt2 is not None) else None
        d_hw = (hw2 - hw1) if (hw1 is not None and hw2 is not None) else None
        max_abs = max([abs(x) for x in (d_alt, d_hw) if x is not None] or [-1])
        h2h.append({
            "task": r["task"],
            "alt1": alt1, "alt2": alt2, "d_alt": d_alt,
            "hw1": hw1, "hw2": hw2, "d_hw": d_hw,
            "max_abs": max_abs,
        })
    h2h.sort(key=lambda x: x["max_abs"], reverse=True)

    # 1m aggregate
    def mean(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        return sum(xs) / len(xs)

    # Common-tasks basis: tasks where BOTH m1 and m2 variant present for the respective pair
    alt_common = [(r["alt1"], r["alt2"]) for r in h2h if r["alt1"] is not None and r["alt2"] is not None]
    hw_common = [(r["hw1"], r["hw2"]) for r in h2h if r["hw1"] is not None and r["hw2"] is not None]
    alt_m1_mean = mean([x[0] for x in alt_common])
    alt_m2_mean = mean([x[1] for x in alt_common])
    hw_m1_mean = mean([x[0] for x in hw_common])
    hw_m2_mean = mean([x[1] for x in hw_common])
    alt_wins = sum(1 for a, b in alt_common if b > a)
    hw_wins = sum(1 for a, b in hw_common if b > a)

    # context archs over all tasks (keep full coverage number per-arch)
    def agg_for(arch):
        vals = [r[arch] for r in rows_1m if r[arch] is not None]
        return len(vals), (sum(vals) / len(vals) if vals else None)

    n_alt1, m_alt1 = agg_for("alt_attn_mamba")
    n_alt2, m_alt2 = agg_for("alt_attn_mamba2")
    n_hw1, m_hw1 = agg_for("headwise")
    n_hw2, m_hw2 = agg_for("headwise_mamba2")
    n_attn, m_attn = agg_for("attn")
    n_mam, m_mam = agg_for("mamba")
    n_mam2, m_mam2 = agg_for("mamba2")

    # 10m aggregate
    def agg_10m(arch):
        vals = [r[arch] for r in rows_10m if r[arch] is not None]
        return len(vals), (sum(vals) / len(vals) if vals else None)

    n10 = {a: agg_10m(a) for a in archs_10m}

    # Anomaly detection
    anomalies = []
    for store_name, store in [("hybrids", hybrids), ("baselines", baselines), ("sweep10", sweep10)]:
        for key, rec in store.items():
            if rec["nan"]:
                anomalies.append((store_name, key, "NaN in tok_acc / eval_loss"))
            elif rec["final_tok_acc"] is not None and rec["final_tok_acc"] <= 0.01:
                anomalies.append((store_name, key, f"collapsed tok_acc={rec['final_tok_acc']:.3f}"))

    # param count summary per arch @ 1m and 10m (median)
    def params_by_arch(rows, arch):
        vals = [r[f"{arch}__n_params"] for r in rows if r.get(f"{arch}__n_params") is not None]
        if not vals:
            return None
        vals = sorted(vals)
        return vals[len(vals) // 2]

    params_1m = {a: params_by_arch(rows_1m, a) for a in archs_1m}
    params_10m = {a: params_by_arch(rows_10m, a) for a in archs_10m}

    # ---------- write markdown
    lines: list[str] = []
    lines.append("# Mamba-2 hybrids vs Mamba-1 / Mamba-2 / attn baselines")
    lines.append("")
    lines.append(f"Run date: 2026-04-23. Data dirs: `{HYBRIDS_DIR}`, `{BASELINES_OLD}`, `{BASELINES_NEW}`, `{SWEEP_10M}`.")
    lines.append("")

    # Summary
    lines.append("## 1. Summary")
    lines.append("")

    def pct(n, d):
        return f"{n}/{d}"

    lines.append(
        f"- **alt_attn hybrid:** on {len(alt_common)} tasks with both variants, "
        f"mean tok_acc {alt_m1_mean:.3f} (mamba-1) vs {alt_m2_mean:.3f} (mamba-2), "
        f"Δmean = {fmt_delta(alt_m2_mean - alt_m1_mean)}. "
        f"Mamba-2 wins on {pct(alt_wins, len(alt_common))} tasks."
    )
    lines.append(
        f"- **headwise hybrid:** on {len(hw_common)} tasks with both variants, "
        f"mean tok_acc {hw_m1_mean:.3f} (mamba-1) vs {hw_m2_mean:.3f} (mamba-2), "
        f"Δmean = {fmt_delta(hw_m2_mean - hw_m1_mean)}. "
        f"Mamba-2 wins on {pct(hw_wins, len(hw_common))} tasks."
    )
    # biggest regressions / improvements
    reg_alt = sorted([r for r in h2h if r["d_alt"] is not None], key=lambda r: r["d_alt"])
    reg_hw = sorted([r for r in h2h if r["d_hw"] is not None], key=lambda r: r["d_hw"])
    if reg_alt:
        worst = reg_alt[0]
        best = reg_alt[-1]
        lines.append(
            f"- **alt_attn extremes:** biggest regression `{worst['task']}` "
            f"{fmt(worst['alt1'])}→{fmt(worst['alt2'])} ({fmt_delta(worst['d_alt'])}); "
            f"biggest gain `{best['task']}` "
            f"{fmt(best['alt1'])}→{fmt(best['alt2'])} ({fmt_delta(best['d_alt'])})."
        )
    if reg_hw:
        worst = reg_hw[0]
        best = reg_hw[-1]
        lines.append(
            f"- **headwise extremes:** biggest regression `{worst['task']}` "
            f"{fmt(worst['hw1'])}→{fmt(worst['hw2'])} ({fmt_delta(worst['d_hw'])}); "
            f"biggest gain `{best['task']}` "
            f"{fmt(best['hw1'])}→{fmt(best['hw2'])} ({fmt_delta(best['d_hw'])})."
        )
    # 10m absolute summary
    if n10["attn"][1] is not None:
        lines.append(
            f"- **10m absolute (mean tok_acc):** attn {n10['attn'][1]:.3f}, "
            f"alt_attn_mamba2 {n10['alt_attn_mamba2'][1]:.3f}, "
            f"headwise_mamba2 {n10['headwise_mamba2'][1]:.3f}, "
            f"lstm {n10['lstm'][1]:.3f}, stu {n10['stu'][1]:.3f}."
        )

    lines.append("")

    # Section 2: head-to-head
    lines.append("## 2. 1m head-to-head: Mamba-1 vs Mamba-2 hybrids")
    lines.append("")
    lines.append("Final `tok_acc`. Sorted by max |Δ| across the two hybrid pairs, descending.")
    lines.append("Bold entries have |Δ| ≥ 0.1 (the new value is bolded).")
    lines.append("")
    lines.append("| task | alt_attn_mamba_1 | alt_attn_mamba_2 | Δ alt | headwise_mamba_1 | headwise_mamba_2 | Δ hw |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in h2h:
        alt2_str = highlight(r["d_alt"], fmt(r["alt2"]))
        hw2_str = highlight(r["d_hw"], fmt(r["hw2"]))
        lines.append(
            f"| {r['task']} | {fmt(r['alt1'])} | {alt2_str} | {fmt_delta(r['d_alt'])} "
            f"| {fmt(r['hw1'])} | {hw2_str} | {fmt_delta(r['d_hw'])} |"
        )
    lines.append("")

    # Section 3: aggregates
    lines.append("## 3. 1m aggregates")
    lines.append("")
    lines.append("| arch | n_tasks | mean tok_acc |")
    lines.append("|---|---:|---:|")
    for arch, (n, m) in [
        ("alt_attn_mamba (m1)", (n_alt1, m_alt1)),
        ("alt_attn_mamba2 (m2)", (n_alt2, m_alt2)),
        ("headwise (m1)", (n_hw1, m_hw1)),
        ("headwise_mamba2 (m2)", (n_hw2, m_hw2)),
        ("attn", (n_attn, m_attn)),
        ("mamba", (n_mam, m_mam)),
        ("mamba2", (n_mam2, m_mam2)),
    ]:
        lines.append(f"| {arch} | {n} | {fmt(m)} |")
    lines.append("")
    lines.append(
        f"On common-task subsets: alt-hybrid mamba-2 wins {alt_wins}/{len(alt_common)} tasks, "
        f"headwise-hybrid mamba-2 wins {hw_wins}/{len(hw_common)} tasks over their mamba-1 counterparts."
    )
    lines.append("")

    # Section 4: 10m absolute
    lines.append("## 4. 10m absolute")
    lines.append("")
    lines.append("Final `tok_acc`. Flagged `*` when `alt_attn_mamba2` or `headwise_mamba2` beats `attn` on that task.")
    lines.append("")
    lines.append("| task | attn_10m | alt_attn_mamba2_10m | headwise_mamba2_10m | lstm_10m | stu_10m |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows_10m:
        attn = r["attn"]
        alt = r["alt_attn_mamba2"]
        hw = r["headwise_mamba2"]
        beats = False
        if attn is not None:
            if (alt is not None and alt > attn) or (hw is not None and hw > attn):
                beats = True
        flag = " *" if beats else ""
        lines.append(
            f"| {r['task']}{flag} | {fmt(attn)} | {fmt(alt)} | {fmt(hw)} | {fmt(r['lstm'])} | {fmt(r['stu'])} |"
        )
    lines.append("")

    # Section 5: 10m aggregate
    lines.append("## 5. 10m aggregates")
    lines.append("")
    lines.append("| arch | n_tasks | mean tok_acc |")
    lines.append("|---|---:|---:|")
    for a in archs_10m:
        n, m = n10[a]
        lines.append(f"| {a} | {n} | {fmt(m)} |")
    lines.append("")

    # Section 6: notes
    lines.append("## 6. Notes")
    lines.append("")
    lines.append("### Param counts (median per arch)")
    lines.append("")
    lines.append("| arch | params @ 1m | params @ 10m |")
    lines.append("|---|---:|---:|")
    for a in archs_1m:
        p1 = params_1m.get(a)
        p10 = params_10m.get(a) if a in archs_10m else None
        p1s = f"{p1:,}" if p1 else "—"
        p10s = f"{p10:,}" if p10 else "—"
        lines.append(f"| {a} | {p1s} | {p10s} |")
    for a in archs_10m:
        if a in archs_1m:
            continue
        p10 = params_10m.get(a)
        p10s = f"{p10:,}" if p10 else "—"
        lines.append(f"| {a} | — | {p10s} |")
    lines.append("")

    lines.append("### Anomalies / skipped runs")
    lines.append("")
    if not anomalies:
        lines.append("No NaN losses or fully-collapsed (`tok_acc ≤ 0.01`) runs detected across any of the four stores.")
    else:
        lines.append("| store | task | arch | scale | issue |")
        lines.append("|---|---|---|---|---|")
        for store_name, key, msg in anomalies:
            task, arch, scale = key
            lines.append(f"| {store_name} | {task} | {arch} | {scale} | {msg} |")
    lines.append("")

    # Coverage notes
    missing_alt = [t for t in all_tasks_1m if rows_1m_byname(rows_1m, t)["alt_attn_mamba"] is None or rows_1m_byname(rows_1m, t)["alt_attn_mamba2"] is None]
    missing_hw = [t for t in all_tasks_1m if rows_1m_byname(rows_1m, t)["headwise"] is None or rows_1m_byname(rows_1m, t)["headwise_mamba2"] is None]
    lines.append("### Coverage gaps at 1m")
    lines.append("")
    lines.append(f"- Tasks missing at least one of `alt_attn_mamba` or `alt_attn_mamba2`: {len(missing_alt)}"
                 + (f" → {', '.join(missing_alt)}" if missing_alt else ""))
    lines.append(f"- Tasks missing at least one of `headwise` or `headwise_mamba2`: {len(missing_hw)}"
                 + (f" → {', '.join(missing_hw)}" if missing_hw else ""))
    lines.append("")
    lines.append(f"CSVs: `{OUT_1M}`, `{OUT_10M}`.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_1M} ({len(rows_1m)} rows)")
    print(f"wrote {OUT_10M} ({len(rows_10m)} rows)")


def rows_1m_byname(rows, task):
    for r in rows:
        if r["task"] == task:
            return r
    return {}


if __name__ == "__main__":
    main()
