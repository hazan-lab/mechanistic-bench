"""Build the alt_attn_stu vs siblings comparison report.

Writes:
- /home/tt6444/mechanistic-bench/figures/alt_attn_stu_1m.csv
- /home/tt6444/mechanistic-bench/figures/alt_attn_stu_10m.csv
- /home/tt6444/mechanistic-bench/figures/alt_attn_stu_report.md
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ALT_STU_DIR = Path("/scratch/gpfs/EHAZAN/tharuntk/mech_runs/alt_attn_stu")
HYBRIDS_DIR = Path("/scratch/gpfs/EHAZAN/tharuntk/mech_runs/mamba2_hybrids")
BASELINES_OLD = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs")
BASELINES_NEW = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs_20260420_212136")
SWEEP_10M = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/sweep_10m_hard")

OUT_DIR = Path("/home/tt6444/mechanistic-bench/figures")
OUT_1M = OUT_DIR / "alt_attn_stu_1m.csv"
OUT_10M = OUT_DIR / "alt_attn_stu_10m.csv"
OUT_MD = OUT_DIR / "alt_attn_stu_report.md"

KNOWN_ARCHS = [
    "alt_attn_mamba2",
    "alt_attn_mamba",
    "alt_attn_stu",
    "headwise_mamba2",
    "headwise_stu",
    "headwise",
    "mamba2",
    "mamba",
    "attn",
    "lstm",
    "stu",
]


def parse_run_name(name: str) -> tuple[str, str, str] | None:
    rest = name
    if rest.endswith("-hard"):
        rest = rest[: -len("-hard")]
    if rest.endswith("-1m"):
        scale = "1m"
        rest = rest[: -len("-1m")]
    elif rest.endswith("-10m"):
        scale = "10m"
        rest = rest[: -len("-10m")]
    else:
        return None
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
    out: dict[tuple[str, str, str], dict] = {}
    for base in dirs:
        if not base.exists():
            continue
        for sub in base.iterdir():
            if not sub.is_dir():
                continue
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

    alt_stu = collect([ALT_STU_DIR])
    hybrids = collect([HYBRIDS_DIR])
    baselines = collect([BASELINES_OLD, BASELINES_NEW])
    sweep10 = collect([SWEEP_10M])

    def lookup_any(stores, task, arch, scale):
        for s in stores:
            rec = s.get((task, arch, scale))
            if rec is not None:
                return rec
        return None

    # ------------------------------------------------------------------ 1m
    archs_1m = ["attn", "alt_attn_mamba", "alt_attn_mamba2", "alt_attn_stu", "stu"]
    # source ordering preference:
    src_map_1m = {
        "attn": [baselines],
        "alt_attn_mamba": [baselines],  # union old+new with tie-break newer mtime (already done in collect)
        "alt_attn_mamba2": [hybrids],
        "alt_attn_stu": [alt_stu],
        "stu": [baselines],
    }
    all_tasks_1m = sorted({
        t for (t, a, s) in list(alt_stu.keys()) if s == "1m" and a == "alt_attn_stu"
    })

    rows_1m = []
    for task in all_tasks_1m:
        row = {"task": task}
        for arch in archs_1m:
            rec = lookup_any(src_map_1m[arch], task, arch, "1m")
            row[arch] = rec["final_tok_acc"] if rec and not rec["nan"] else None
            row[f"{arch}__n_params"] = rec["n_params"] if rec else None
            row[f"{arch}__nan"] = rec["nan"] if rec else False
        rows_1m.append(row)

    with OUT_1M.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task"] + archs_1m)
        for r in rows_1m:
            w.writerow([r["task"]] + [r[a] for a in archs_1m])

    # ------------------------------------------------------------------ 10m
    archs_10m = ["attn", "alt_attn_mamba2", "alt_attn_stu", "stu", "lstm"]
    src_map_10m = {
        "attn": [sweep10],
        "alt_attn_mamba2": [hybrids],
        "alt_attn_stu": [alt_stu],
        "stu": [sweep10],
        "lstm": [sweep10],
    }
    all_tasks_10m = sorted({
        t for (t, a, s) in list(alt_stu.keys()) if s == "10m" and a == "alt_attn_stu"
    })

    rows_10m = []
    for task in all_tasks_10m:
        row = {"task": task}
        for arch in archs_10m:
            rec = lookup_any(src_map_10m[arch], task, arch, "10m")
            row[arch] = rec["final_tok_acc"] if rec and not rec["nan"] else None
            row[f"{arch}__n_params"] = rec["n_params"] if rec else None
            row[f"{arch}__nan"] = rec["nan"] if rec else False
        rows_10m.append(row)

    with OUT_10M.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task"] + archs_10m)
        for r in rows_10m:
            w.writerow([r["task"]] + [r[a] for a in archs_10m])

    # --- helpers
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

    def mean(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        return sum(xs) / len(xs)

    # --- 1m h2h: sort by alt_attn_stu - alt_attn_mamba desc
    for r in rows_1m:
        a_stu = r["alt_attn_stu"]
        a_m1 = r["alt_attn_mamba"]
        a_m2 = r["alt_attn_mamba2"]
        p_attn = r["attn"]
        p_stu = r["stu"]
        r["d_stu_vs_m1"] = (a_stu - a_m1) if (a_stu is not None and a_m1 is not None) else None
        r["d_stu_vs_m2"] = (a_stu - a_m2) if (a_stu is not None and a_m2 is not None) else None
        r["d_stu_vs_attn"] = (a_stu - p_attn) if (a_stu is not None and p_attn is not None) else None
        r["d_stu_vs_pstu"] = (a_stu - p_stu) if (a_stu is not None and p_stu is not None) else None

    def sort_key_1m(r):
        d = r["d_stu_vs_m1"]
        return (-(d if d is not None else -1e9),)

    rows_1m_sorted = sorted(rows_1m, key=sort_key_1m)

    # Intersection means: tasks where every 1m column present
    inter_1m = [
        r for r in rows_1m
        if all(r[a] is not None for a in archs_1m)
    ]
    means_1m_inter = {a: mean([r[a] for r in inter_1m]) for a in archs_1m}

    def win_count(rows, a_new, a_ref):
        pairs = [(r[a_new], r[a_ref]) for r in rows if r[a_new] is not None and r[a_ref] is not None]
        wins = sum(1 for x, y in pairs if x > y)
        ties = sum(1 for x, y in pairs if x == y)
        return wins, len(pairs), ties

    w_vs_m1 = win_count(rows_1m, "alt_attn_stu", "alt_attn_mamba")
    w_vs_m2 = win_count(rows_1m, "alt_attn_stu", "alt_attn_mamba2")
    w_vs_pstu = win_count(rows_1m, "alt_attn_stu", "stu")
    w_vs_attn = win_count(rows_1m, "alt_attn_stu", "attn")

    def coverage(rows, a):
        return sum(1 for r in rows if r[a] is not None)

    cov_1m = {a: coverage(rows_1m, a) for a in archs_1m}
    means_1m_all = {a: mean([r[a] for r in rows_1m]) for a in archs_1m}

    # --- 10m
    for r in rows_10m:
        a_stu = r["alt_attn_stu"]
        a_m2 = r["alt_attn_mamba2"]
        p_attn = r["attn"]
        p_stu = r["stu"]
        p_lstm = r["lstm"]
        r["d_stu_vs_m2"] = (a_stu - a_m2) if (a_stu is not None and a_m2 is not None) else None
        r["d_stu_vs_attn"] = (a_stu - p_attn) if (a_stu is not None and p_attn is not None) else None
        r["d_stu_vs_pstu"] = (a_stu - p_stu) if (a_stu is not None and p_stu is not None) else None
        r["d_stu_vs_lstm"] = (a_stu - p_lstm) if (a_stu is not None and p_lstm is not None) else None

    rows_10m_sorted = sorted(
        rows_10m,
        key=lambda r: -((r["d_stu_vs_m2"] if r["d_stu_vs_m2"] is not None else -1e9)),
    )

    inter_10m = [r for r in rows_10m if all(r[a] is not None for a in archs_10m)]
    means_10m_inter = {a: mean([r[a] for r in inter_10m]) for a in archs_10m}
    means_10m_all = {a: mean([r[a] for r in rows_10m]) for a in archs_10m}
    cov_10m = {a: coverage(rows_10m, a) for a in archs_10m}

    w10_vs_m2 = win_count(rows_10m, "alt_attn_stu", "alt_attn_mamba2")
    w10_vs_attn = win_count(rows_10m, "alt_attn_stu", "attn")
    w10_vs_pstu = win_count(rows_10m, "alt_attn_stu", "stu")
    w10_vs_lstm = win_count(rows_10m, "alt_attn_stu", "lstm")

    # extremes
    with_dm1 = [r for r in rows_1m if r["d_stu_vs_m1"] is not None]
    with_dm1_sorted = sorted(with_dm1, key=lambda r: r["d_stu_vs_m1"])
    biggest_loss_vs_m1 = with_dm1_sorted[0] if with_dm1_sorted else None
    biggest_gain_vs_m1 = with_dm1_sorted[-1] if with_dm1_sorted else None

    with_dattn = [r for r in rows_1m if r["d_stu_vs_attn"] is not None]
    with_dattn_sorted = sorted(with_dattn, key=lambda r: r["d_stu_vs_attn"])
    biggest_loss_vs_attn = with_dattn_sorted[0] if with_dattn_sorted else None
    biggest_gain_vs_attn = with_dattn_sorted[-1] if with_dattn_sorted else None

    with_dpstu = [r for r in rows_1m if r["d_stu_vs_pstu"] is not None]
    with_dpstu_sorted = sorted(with_dpstu, key=lambda r: r["d_stu_vs_pstu"])

    # --- write markdown
    lines: list[str] = []
    lines.append("# alt_attn_stu (layer-wise attn+STU hybrid) vs siblings")
    lines.append("")
    lines.append(
        f"Run date: 2026-04-23. Data dirs: `{ALT_STU_DIR}`, `{HYBRIDS_DIR}`, "
        f"`{BASELINES_OLD}`, `{BASELINES_NEW}`, `{SWEEP_10M}`."
    )
    lines.append("")

    # 1. Summary
    lines.append("## 1. Summary")
    lines.append("")

    def wl(w, n, t):
        # win / total (ties counted separately if any)
        if t:
            return f"{w}/{n} (ties {t})"
        return f"{w}/{n}"

    # Aggregate intersection means
    m_stu = means_1m_inter.get("alt_attn_stu")
    m_m1 = means_1m_inter.get("alt_attn_mamba")
    m_m2 = means_1m_inter.get("alt_attn_mamba2")
    m_attn = means_1m_inter.get("attn")
    m_pstu = means_1m_inter.get("stu")

    lines.append(
        f"- **1m intersection ({len(inter_1m)} tasks), mean tok_acc:** "
        f"attn {fmt(m_attn)} | alt_attn_mamba {fmt(m_m1)} | "
        f"alt_attn_mamba2 {fmt(m_m2)} | **alt_attn_stu {fmt(m_stu)}** | pure stu {fmt(m_pstu)}."
    )
    # Headline comparisons
    if m_stu is not None and m_m1 is not None:
        lines.append(
            f"- **alt_attn_stu vs alt_attn_mamba (prior-best hybrid):** "
            f"Δmean {fmt_delta(m_stu - m_m1)}; wins {wl(*w_vs_m1)} tasks."
        )
    if m_stu is not None and m_attn is not None:
        lines.append(
            f"- **alt_attn_stu vs pure attn:** Δmean {fmt_delta(m_stu - m_attn)}; "
            f"wins {wl(*w_vs_attn)} tasks."
        )
    if m_stu is not None and m_pstu is not None:
        lines.append(
            f"- **alt_attn_stu vs pure stu:** Δmean {fmt_delta(m_stu - m_pstu)}; "
            f"wins {wl(*w_vs_pstu)} tasks."
        )
    if m_stu is not None and m_m2 is not None:
        lines.append(
            f"- **alt_attn_stu vs alt_attn_mamba2:** Δmean {fmt_delta(m_stu - m_m2)}; "
            f"wins {wl(*w_vs_m2)} tasks."
        )
    # 10m headline
    m10_stu = means_10m_inter.get("alt_attn_stu")
    m10_attn = means_10m_inter.get("attn")
    m10_m2 = means_10m_inter.get("alt_attn_mamba2")
    m10_pstu = means_10m_inter.get("stu")
    m10_lstm = means_10m_inter.get("lstm")
    if m10_stu is not None:
        lines.append(
            f"- **10m intersection ({len(inter_10m)} tasks):** "
            f"attn {fmt(m_attn if m_attn is None else m10_attn)} | "
            f"alt_attn_mamba2 {fmt(m10_m2)} | **alt_attn_stu {fmt(m10_stu)}** | "
            f"pure stu {fmt(m10_pstu)} | lstm {fmt(m10_lstm)}. "
            f"(No alt_attn_mamba at 10m.)"
        )
    lines.append("")

    # 2. 1m head-to-head
    lines.append("## 2. 1m head-to-head (per task)")
    lines.append("")
    lines.append(
        "Columns: `attn | alt_attn_mamba | alt_attn_mamba2 | alt_attn_stu | pure stu`. "
        "Δ columns in the right block: `alt_attn_stu − alt_attn_mamba` (sort key) and "
        "`alt_attn_stu − attn`. Sorted by `alt_attn_stu − alt_attn_mamba` desc. "
        "Bold entries have |Δ vs alt_attn_mamba| ≥ 0.1."
    )
    lines.append("")
    lines.append(
        "| task | attn | alt_attn_mamba | alt_attn_mamba2 | alt_attn_stu | pure stu | Δ stu−m1 | Δ stu−attn |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows_1m_sorted:
        stu_str = highlight(r["d_stu_vs_m1"], fmt(r["alt_attn_stu"]))
        lines.append(
            f"| {r['task']} | {fmt(r['attn'])} | {fmt(r['alt_attn_mamba'])} | "
            f"{fmt(r['alt_attn_mamba2'])} | {stu_str} | {fmt(r['stu'])} | "
            f"{fmt_delta(r['d_stu_vs_m1'])} | {fmt_delta(r['d_stu_vs_attn'])} |"
        )
    lines.append("")

    # 3. 1m aggregates
    lines.append("## 3. 1m aggregates")
    lines.append("")
    lines.append(f"**Intersection basis** ({len(inter_1m)} tasks with all five columns):")
    lines.append("")
    lines.append("| arch | n | mean tok_acc |")
    lines.append("|---|---:|---:|")
    for a in archs_1m:
        lines.append(f"| {a} | {len(inter_1m)} | {fmt(means_1m_inter.get(a))} |")
    lines.append("")
    lines.append("**All-task basis** (per-arch coverage, mean over available tasks):")
    lines.append("")
    lines.append("| arch | n_tasks | mean tok_acc |")
    lines.append("|---|---:|---:|")
    for a in archs_1m:
        lines.append(f"| {a} | {cov_1m[a]} | {fmt(means_1m_all.get(a))} |")
    lines.append("")
    lines.append("**Win counts (alt_attn_stu beats X on pairwise-present tasks):**")
    lines.append("")
    lines.append("| comparison | wins / n |")
    lines.append("|---|---:|")
    lines.append(f"| alt_attn_stu vs alt_attn_mamba | {wl(*w_vs_m1)} |")
    lines.append(f"| alt_attn_stu vs alt_attn_mamba2 | {wl(*w_vs_m2)} |")
    lines.append(f"| alt_attn_stu vs pure stu | {wl(*w_vs_pstu)} |")
    lines.append(f"| alt_attn_stu vs pure attn | {wl(*w_vs_attn)} |")
    lines.append("")

    # Extremes
    lines.append("**Extremes at 1m:**")
    lines.append("")
    if biggest_gain_vs_m1 is not None:
        r = biggest_gain_vs_m1
        lines.append(
            f"- Biggest gain vs alt_attn_mamba: `{r['task']}` "
            f"{fmt(r['alt_attn_mamba'])} → {fmt(r['alt_attn_stu'])} ({fmt_delta(r['d_stu_vs_m1'])})."
        )
    if biggest_loss_vs_m1 is not None:
        r = biggest_loss_vs_m1
        lines.append(
            f"- Biggest regression vs alt_attn_mamba: `{r['task']}` "
            f"{fmt(r['alt_attn_mamba'])} → {fmt(r['alt_attn_stu'])} ({fmt_delta(r['d_stu_vs_m1'])})."
        )
    if biggest_gain_vs_attn is not None:
        r = biggest_gain_vs_attn
        lines.append(
            f"- Biggest gain vs pure attn: `{r['task']}` "
            f"{fmt(r['attn'])} → {fmt(r['alt_attn_stu'])} ({fmt_delta(r['d_stu_vs_attn'])})."
        )
    if biggest_loss_vs_attn is not None:
        r = biggest_loss_vs_attn
        lines.append(
            f"- Biggest regression vs pure attn: `{r['task']}` "
            f"{fmt(r['attn'])} → {fmt(r['alt_attn_stu'])} ({fmt_delta(r['d_stu_vs_attn'])})."
        )
    if with_dpstu_sorted:
        r = with_dpstu_sorted[-1]
        lines.append(
            f"- Biggest gain vs pure stu: `{r['task']}` "
            f"{fmt(r['stu'])} → {fmt(r['alt_attn_stu'])} ({fmt_delta(r['d_stu_vs_pstu'])})."
        )
        r = with_dpstu_sorted[0]
        lines.append(
            f"- Biggest regression vs pure stu: `{r['task']}` "
            f"{fmt(r['stu'])} → {fmt(r['alt_attn_stu'])} ({fmt_delta(r['d_stu_vs_pstu'])})."
        )
    lines.append("")

    # 4. 10m h2h
    lines.append("## 4. 10m head-to-head (per task)")
    lines.append("")
    lines.append(
        "Columns: `attn | alt_attn_mamba2 | alt_attn_stu | pure stu | lstm`. "
        "Sorted by `alt_attn_stu − alt_attn_mamba2` desc. "
        "Bold entries have |Δ vs alt_attn_mamba2| ≥ 0.1. "
        "No `alt_attn_mamba` (Mamba-1) runs exist at 10m."
    )
    lines.append("")
    lines.append(
        "| task | attn | alt_attn_mamba2 | alt_attn_stu | pure stu | lstm | Δ stu−m2 | Δ stu−attn |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows_10m_sorted:
        stu_str = highlight(r["d_stu_vs_m2"], fmt(r["alt_attn_stu"]))
        lines.append(
            f"| {r['task']} | {fmt(r['attn'])} | {fmt(r['alt_attn_mamba2'])} | "
            f"{stu_str} | {fmt(r['stu'])} | {fmt(r['lstm'])} | "
            f"{fmt_delta(r['d_stu_vs_m2'])} | {fmt_delta(r['d_stu_vs_attn'])} |"
        )
    lines.append("")

    # 5. 10m aggregates
    lines.append("## 5. 10m aggregates")
    lines.append("")
    lines.append(f"**Intersection basis** ({len(inter_10m)} tasks with all five columns):")
    lines.append("")
    lines.append("| arch | n | mean tok_acc |")
    lines.append("|---|---:|---:|")
    for a in archs_10m:
        lines.append(f"| {a} | {len(inter_10m)} | {fmt(means_10m_inter.get(a))} |")
    lines.append("")
    lines.append("**All-task basis:**")
    lines.append("")
    lines.append("| arch | n_tasks | mean tok_acc |")
    lines.append("|---|---:|---:|")
    for a in archs_10m:
        lines.append(f"| {a} | {cov_10m[a]} | {fmt(means_10m_all.get(a))} |")
    lines.append("")
    lines.append("**Win counts:**")
    lines.append("")
    lines.append("| comparison | wins / n |")
    lines.append("|---|---:|")
    lines.append(f"| alt_attn_stu vs alt_attn_mamba2 | {wl(*w10_vs_m2)} |")
    lines.append(f"| alt_attn_stu vs pure attn | {wl(*w10_vs_attn)} |")
    lines.append(f"| alt_attn_stu vs pure stu | {wl(*w10_vs_pstu)} |")
    lines.append(f"| alt_attn_stu vs lstm | {wl(*w10_vs_lstm)} |")
    lines.append("")

    # 6. Takeaway
    lines.append("## 6. Takeaway")
    lines.append("")
    # Build concrete sentences based on numbers
    bullet = []
    # vs attn
    if m_stu is not None and m_attn is not None:
        d = m_stu - m_attn
        w, n, _ = w_vs_attn
        if d > 0:
            bullet.append(
                f"- At 1m, alt_attn_stu beats pure attn by {fmt_delta(d)} on the "
                f"{len(inter_1m)}-task intersection (wins {w}/{n} pairwise). "
                "Layer-wise attn+X does yield a recipe that beats pure attn on mech-bench, "
                "and STU is a viable alternating branch."
            )
        else:
            bullet.append(
                f"- At 1m, alt_attn_stu does not beat pure attn (Δmean {fmt_delta(d)}, "
                f"pairwise wins {w}/{n}). No layer-wise attn+X recipe tested here decisively beats pure attn."
            )
    # vs alt_attn_mamba
    if m_stu is not None and m_m1 is not None:
        d = m_stu - m_m1
        w, n, _ = w_vs_m1
        if d > 0:
            bullet.append(
                f"- alt_attn_stu outperforms alt_attn_mamba (prior-best hybrid) by {fmt_delta(d)} mean, "
                f"winning {w}/{n} tasks — STU is stronger than Mamba-1 as the non-attn branch at 1m."
            )
        else:
            bullet.append(
                f"- alt_attn_stu underperforms alt_attn_mamba (Mamba-1) by {fmt_delta(d)} mean "
                f"(wins {w}/{n}). The layer-wise recipe's benefit at 1m is specific to Mamba-1, not generic across branches."
            )
    # vs pure stu
    if m_stu is not None and m_pstu is not None:
        d = m_stu - m_pstu
        w, n, _ = w_vs_pstu
        bullet.append(
            f"- Adding alternating attention layers to STU is {('helpful' if d > 0 else 'not helpful')}: "
            f"Δmean {fmt_delta(d)}, wins {w}/{n} vs pure stu."
        )
    # 10m
    if m10_stu is not None and m10_attn is not None:
        d10 = m10_stu - m10_attn
        w, n, _ = w10_vs_attn
        bullet.append(
            f"- At 10m, alt_attn_stu vs pure attn: Δmean {fmt_delta(d10)}, wins {w}/{n}."
        )
    if m10_stu is not None and m10_m2 is not None:
        d10 = m10_stu - m10_m2
        w, n, _ = w10_vs_m2
        bullet.append(
            f"- At 10m, alt_attn_stu vs alt_attn_mamba2: Δmean {fmt_delta(d10)}, wins {w}/{n}."
        )
    # Standouts
    if biggest_gain_vs_m1 is not None and biggest_loss_vs_m1 is not None:
        bullet.append(
            f"- Standout STU-helps tasks (vs alt_attn_mamba): "
            f"`{biggest_gain_vs_m1['task']}` ({fmt_delta(biggest_gain_vs_m1['d_stu_vs_m1'])}). "
            f"Standout STU-hurts tasks: "
            f"`{biggest_loss_vs_m1['task']}` ({fmt_delta(biggest_loss_vs_m1['d_stu_vs_m1'])})."
        )
    lines.extend(bullet)
    lines.append("")
    lines.append(f"CSVs: `{OUT_1M}`, `{OUT_10M}`.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_1M} ({len(rows_1m)} rows)")
    print(f"wrote {OUT_10M} ({len(rows_10m)} rows)")

    # print key stats for the caller
    print()
    print("=== KEY STATS ===")
    print(f"1m intersection tasks: {len(inter_1m)}")
    print(f"  means: attn={fmt(means_1m_inter.get('attn'))} alt_m1={fmt(means_1m_inter.get('alt_attn_mamba'))} "
          f"alt_m2={fmt(means_1m_inter.get('alt_attn_mamba2'))} alt_stu={fmt(means_1m_inter.get('alt_attn_stu'))} "
          f"pstu={fmt(means_1m_inter.get('stu'))}")
    print(f"  wins vs m1: {w_vs_m1}, vs m2: {w_vs_m2}, vs pstu: {w_vs_pstu}, vs attn: {w_vs_attn}")
    print(f"1m coverage: {cov_1m}")
    print(f"10m intersection tasks: {len(inter_10m)}")
    print(f"  means: {{{', '.join(f'{a}={fmt(means_10m_inter.get(a))}' for a in archs_10m)}}}")
    print(f"  wins vs m2: {w10_vs_m2}, vs attn: {w10_vs_attn}, vs pstu: {w10_vs_pstu}, vs lstm: {w10_vs_lstm}")
    print(f"10m coverage: {cov_10m}")


if __name__ == "__main__":
    main()
