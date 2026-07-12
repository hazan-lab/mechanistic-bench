"""Build the headwise_stu (attn+STU) vs sibling variants comparison report.

Siblings:
- headwise (attn+Mamba-1)        -- baseline
- headwise_mamba2 (attn+Mamba-2) -- previous attempt
- stu (pure STU)                 -- no-attention baseline
- attn                           -- pure attention (context)
- lstm (10m only)                -- extra context

Writes:
- /home/tt6444/mechanistic-bench/figures/headwise_stu_1m.csv
- /home/tt6444/mechanistic-bench/figures/headwise_stu_10m.csv
- /home/tt6444/mechanistic-bench/figures/headwise_stu_report.md
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HEADWISE_STU_DIR = Path("/scratch/gpfs/EHAZAN/tharuntk/mech_runs/headwise_stu")
HYBRIDS_DIR = Path("/scratch/gpfs/EHAZAN/tharuntk/mech_runs/mamba2_hybrids")
BASELINES_OLD = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs")
BASELINES_NEW = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs_20260420_212136")
SWEEP_10M = Path("/scratch/gpfs/EHAZAN/tharuntk/mechbench/sweep_10m_hard")

OUT_DIR = Path("/home/tt6444/mechanistic-bench/figures")
OUT_1M = OUT_DIR / "headwise_stu_1m.csv"
OUT_10M = OUT_DIR / "headwise_stu_10m.csv"
OUT_MD = OUT_DIR / "headwise_stu_report.md"

# longest-first so multi-underscore archs win over shorter prefixes
KNOWN_ARCHS = [
    "alt_attn_mamba2",
    "alt_attn_mamba",
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


def fmt(v):
    return "—" if v is None else f"{v:.3f}"


def fmt_delta(v):
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.3f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stu_runs = collect([HEADWISE_STU_DIR])
    hybrids = collect([HYBRIDS_DIR])
    baselines = collect([BASELINES_OLD, BASELINES_NEW])
    sweep10 = collect([SWEEP_10M])

    def lookup(*stores, task, arch, scale):
        for s in stores:
            rec = s.get((task, arch, scale))
            if rec is not None:
                return rec
        return None

    # ---------------- 1m ----------------
    archs_1m = ["headwise", "headwise_mamba2", "headwise_stu", "stu", "attn"]
    tasks_1m = sorted({
        t for store in (stu_runs, hybrids, baselines)
        for (t, a, s) in store if s == "1m" and a in archs_1m
    })

    rows_1m = []
    for task in tasks_1m:
        row = {"task": task}
        for arch in archs_1m:
            rec = lookup(stu_runs, hybrids, baselines, task=task, arch=arch, scale="1m")
            row[arch] = rec["final_tok_acc"] if rec and not rec["nan"] else None
            row[f"{arch}__n_params"] = rec["n_params"] if rec else None
            row[f"{arch}__nan"] = rec["nan"] if rec else False
        rows_1m.append(row)

    with OUT_1M.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task"] + archs_1m)
        for r in rows_1m:
            w.writerow([r["task"]] + [r[a] for a in archs_1m])

    # ---------------- 10m ----------------
    archs_10m = ["headwise_mamba2", "headwise_stu", "stu", "attn", "lstm"]
    tasks_10m = sorted({
        t for store in (stu_runs, hybrids, sweep10)
        for (t, a, s) in store if s == "10m" and a in archs_10m
    })

    rows_10m = []
    for task in tasks_10m:
        row = {"task": task}
        for arch in archs_10m:
            rec = lookup(stu_runs, hybrids, sweep10, task=task, arch=arch, scale="10m")
            row[arch] = rec["final_tok_acc"] if rec and not rec["nan"] else None
            row[f"{arch}__n_params"] = rec["n_params"] if rec else None
            row[f"{arch}__nan"] = rec["nan"] if rec else False
        rows_10m.append(row)

    with OUT_10M.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task"] + archs_10m)
        for r in rows_10m:
            w.writerow([r["task"]] + [r[a] for a in archs_10m])

    # ---------------- 1m analysis ----------------
    def delta(new, old):
        if new is None or old is None:
            return None
        return new - old

    h2h_1m = []
    for r in rows_1m:
        d_vs_m1 = delta(r["headwise_stu"], r["headwise"])
        d_vs_m2 = delta(r["headwise_stu"], r["headwise_mamba2"])
        d_vs_pureSTU = delta(r["headwise_stu"], r["stu"])
        h2h_1m.append({
            **r,
            "d_vs_m1": d_vs_m1,
            "d_vs_m2": d_vs_m2,
            "d_vs_pureSTU": d_vs_pureSTU,
        })

    # sort by d_vs_m1 desc (biggest wins of headwise_stu vs headwise-m1 first)
    h2h_1m_sorted = sorted(h2h_1m, key=lambda r: (r["d_vs_m1"] is None, -(r["d_vs_m1"] if r["d_vs_m1"] is not None else -1e9)))

    # common intersection for aggregates: tasks where ALL four variants present
    shared_1m = [r for r in h2h_1m if all(r[a] is not None for a in ["headwise", "headwise_mamba2", "headwise_stu", "stu"])]
    means_1m_shared = {a: mean([r[a] for r in shared_1m]) for a in archs_1m}

    # wins of headwise_stu
    def wins(rows, new_key, old_key):
        pairs = [(r[new_key], r[old_key]) for r in rows if r[new_key] is not None and r[old_key] is not None]
        w_cnt = sum(1 for a, b in pairs if a > b)
        t_cnt = sum(1 for a, b in pairs if a == b)
        return w_cnt, t_cnt, len(pairs)

    w_vs_m1 = wins(h2h_1m, "headwise_stu", "headwise")
    w_vs_m2 = wins(h2h_1m, "headwise_stu", "headwise_mamba2")
    w_vs_pureSTU = wins(h2h_1m, "headwise_stu", "stu")

    # per-arch mean over whatever is available (coverage number too)
    def agg(rows, arch):
        vals = [r[arch] for r in rows if r[arch] is not None]
        return len(vals), (sum(vals) / len(vals) if vals else None)

    agg_1m = {a: agg(rows_1m, a) for a in archs_1m}

    # ---------------- 10m analysis ----------------
    h2h_10m = []
    for r in rows_10m:
        d_vs_m2 = delta(r["headwise_stu"], r["headwise_mamba2"])
        d_vs_pureSTU = delta(r["headwise_stu"], r["stu"])
        d_vs_attn = delta(r["headwise_stu"], r["attn"])
        h2h_10m.append({
            **r,
            "d_vs_m2": d_vs_m2,
            "d_vs_pureSTU": d_vs_pureSTU,
            "d_vs_attn": d_vs_attn,
        })
    h2h_10m_sorted = sorted(h2h_10m, key=lambda r: (r["d_vs_m2"] is None, -(r["d_vs_m2"] if r["d_vs_m2"] is not None else -1e9)))

    shared_10m = [r for r in h2h_10m if all(r[a] is not None for a in ["headwise_mamba2", "headwise_stu", "stu", "attn"])]
    means_10m_shared = {a: mean([r[a] for r in shared_10m]) for a in archs_10m}
    w10_vs_m2 = wins(h2h_10m, "headwise_stu", "headwise_mamba2")
    w10_vs_pureSTU = wins(h2h_10m, "headwise_stu", "stu")
    w10_vs_attn = wins(h2h_10m, "headwise_stu", "attn")
    agg_10m = {a: agg(rows_10m, a) for a in archs_10m}

    # anomalies
    anomalies = []
    for name, store in [("headwise_stu", stu_runs), ("hybrids", hybrids), ("baselines", baselines), ("sweep10", sweep10)]:
        for key, rec in store.items():
            t, a, s = key
            if a not in set(archs_1m) | set(archs_10m):
                continue
            if rec["nan"]:
                anomalies.append((name, key, "NaN in tok_acc / eval_loss"))
            elif rec["final_tok_acc"] is not None and rec["final_tok_acc"] <= 0.01:
                anomalies.append((name, key, f"collapsed tok_acc={rec['final_tok_acc']:.3f}"))

    # ---------------- markdown ----------------
    lines: list[str] = []
    lines.append("# headwise_stu (attn+STU) vs siblings")
    lines.append("")
    lines.append(
        f"Run date: 2026-04-23. Sources: `{HEADWISE_STU_DIR}`, `{HYBRIDS_DIR}`, "
        f"`{BASELINES_OLD}`, `{BASELINES_NEW}`, `{SWEEP_10M}`."
    )
    lines.append("")
    lines.append(
        "Hypothesis under test: Mamba-2 hurt the headwise hybrid even though pure Mamba-2 beats pure "
        "Mamba-1 — maybe the Mamba kernel is a bad fit for a head-split and STU would pair better "
        "with attention."
    )
    lines.append("")

    # summary
    lines.append("## 1. Summary")
    lines.append("")

    def pct(t):
        w, tied, n = t
        return f"{w}/{n} ({w / n:.0%})" if n else "0/0"

    shared_n = len(shared_1m)
    d_hwstu_m1 = (means_1m_shared["headwise_stu"] - means_1m_shared["headwise"]) if shared_n else None
    d_hwstu_m2 = (means_1m_shared["headwise_stu"] - means_1m_shared["headwise_mamba2"]) if shared_n else None
    d_hwstu_pureSTU = (means_1m_shared["headwise_stu"] - means_1m_shared["stu"]) if shared_n else None

    lines.append(
        f"- **1m shared-task means** (n={shared_n} tasks with all four variants): "
        f"headwise (m1) {fmt(means_1m_shared['headwise'])}, "
        f"headwise_mamba2 {fmt(means_1m_shared['headwise_mamba2'])}, "
        f"**headwise_stu {fmt(means_1m_shared['headwise_stu'])}**, "
        f"pure stu {fmt(means_1m_shared['stu'])}."
    )
    lines.append(
        f"- **headwise_stu vs headwise (m1) baseline:** Δmean tok_acc = {fmt_delta(d_hwstu_m1)}; "
        f"wins on {pct(w_vs_m1)} tasks."
    )
    lines.append(
        f"- **headwise_stu vs headwise_mamba2:** Δmean tok_acc = {fmt_delta(d_hwstu_m2)}; "
        f"wins on {pct(w_vs_m2)} tasks."
    )
    lines.append(
        f"- **headwise_stu vs pure STU:** Δmean tok_acc = {fmt_delta(d_hwstu_pureSTU)}; "
        f"wins on {pct(w_vs_pureSTU)} tasks. "
        f"(If this is ≤ 0, the headwise split is not helping STU — it's just dragging pure STU down.)"
    )
    # biggest extremes
    with_m1 = [r for r in h2h_1m if r["d_vs_m1"] is not None]
    if with_m1:
        best = max(with_m1, key=lambda r: r["d_vs_m1"])
        worst = min(with_m1, key=lambda r: r["d_vs_m1"])
        lines.append(
            f"- **1m extremes vs headwise-m1:** biggest gain `{best['task']}` "
            f"{fmt(best['headwise'])}→{fmt(best['headwise_stu'])} ({fmt_delta(best['d_vs_m1'])}); "
            f"biggest regression `{worst['task']}` "
            f"{fmt(worst['headwise'])}→{fmt(worst['headwise_stu'])} ({fmt_delta(worst['d_vs_m1'])})."
        )
    # 10m one-liner
    if shared_10m:
        lines.append(
            f"- **10m shared-task means** (n={len(shared_10m)}): headwise_mamba2 "
            f"{fmt(means_10m_shared['headwise_mamba2'])}, "
            f"**headwise_stu {fmt(means_10m_shared['headwise_stu'])}**, pure stu "
            f"{fmt(means_10m_shared['stu'])}, attn {fmt(means_10m_shared['attn'])}."
        )
    lines.append(
        "- No `headwise` (Mamba-1) runs exist at 10m — that column is skipped in the 10m table."
    )
    lines.append("")

    # section 2: 1m head-to-head
    lines.append("## 2. 1m head-to-head")
    lines.append("")
    lines.append(
        "Final `tok_acc`. Sorted by (headwise_stu − headwise) descending — top rows are the tasks where "
        "swapping Mamba-1 for STU in the headwise split helped the most."
    )
    lines.append(
        "Bold `headwise_stu` cell: |Δ vs headwise-m1| ≥ 0.1."
    )
    lines.append("")
    lines.append(
        "| task | headwise (m1) | headwise_mamba2 | headwise_stu | Δ vs m1 | Δ vs m2 | pure stu | Δ vs pureSTU | attn |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in h2h_1m_sorted:
        hwstu_str = fmt(r["headwise_stu"])
        if r["d_vs_m1"] is not None and abs(r["d_vs_m1"]) >= 0.1:
            hwstu_str = f"**{hwstu_str}**"
        lines.append(
            f"| {r['task']} | {fmt(r['headwise'])} | {fmt(r['headwise_mamba2'])} | "
            f"{hwstu_str} | {fmt_delta(r['d_vs_m1'])} | {fmt_delta(r['d_vs_m2'])} | "
            f"{fmt(r['stu'])} | {fmt_delta(r['d_vs_pureSTU'])} | {fmt(r['attn'])} |"
        )
    lines.append("")

    # section 3: 1m aggregates
    lines.append("## 3. 1m aggregates")
    lines.append("")
    lines.append("### Per-arch coverage and mean over all tasks present")
    lines.append("")
    lines.append("| arch | n_tasks | mean tok_acc |")
    lines.append("|---|---:|---:|")
    for a in archs_1m:
        n, m = agg_1m[a]
        lines.append(f"| {a} | {n} | {fmt(m)} |")
    lines.append("")
    lines.append(
        f"### Shared-task ({shared_n} tasks, all four of headwise/headwise_mamba2/headwise_stu/stu present)"
    )
    lines.append("")
    lines.append("| arch | mean tok_acc on shared | Δ vs headwise (m1) |")
    lines.append("|---|---:|---:|")
    for a in ["headwise", "headwise_mamba2", "headwise_stu", "stu", "attn"]:
        m_shared = means_1m_shared[a]
        d = (m_shared - means_1m_shared["headwise"]) if (m_shared is not None and means_1m_shared["headwise"] is not None and a != "headwise") else None
        lines.append(f"| {a} | {fmt(m_shared)} | {fmt_delta(d) if a != 'headwise' else '—'} |")
    lines.append("")
    lines.append("### Wins of `headwise_stu`")
    lines.append("")
    lines.append("| opponent | wins / ties / n | win rate |")
    lines.append("|---|---|---:|")
    for label, t in [
        ("headwise (m1)", w_vs_m1),
        ("headwise_mamba2", w_vs_m2),
        ("pure stu", w_vs_pureSTU),
    ]:
        w, tied, n = t
        rate = (w / n) if n else 0.0
        lines.append(f"| {label} | {w} / {tied} / {n} | {rate:.0%} |")
    lines.append("")

    # section 4: 10m head-to-head
    lines.append("## 4. 10m head-to-head")
    lines.append("")
    lines.append(
        "Final `tok_acc`. Sorted by (headwise_stu − headwise_mamba2) descending. "
        "No `headwise` (Mamba-1) at 10m."
    )
    lines.append("")
    lines.append(
        "| task | headwise_mamba2 | headwise_stu | Δ vs m2 | pure stu | Δ vs pureSTU | attn | lstm |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in h2h_10m_sorted:
        hwstu_str = fmt(r["headwise_stu"])
        if r["d_vs_m2"] is not None and abs(r["d_vs_m2"]) >= 0.1:
            hwstu_str = f"**{hwstu_str}**"
        lines.append(
            f"| {r['task']} | {fmt(r['headwise_mamba2'])} | {hwstu_str} | "
            f"{fmt_delta(r['d_vs_m2'])} | {fmt(r['stu'])} | {fmt_delta(r['d_vs_pureSTU'])} | "
            f"{fmt(r['attn'])} | {fmt(r['lstm'])} |"
        )
    lines.append("")

    # section 5: 10m aggregates
    lines.append("## 5. 10m aggregates")
    lines.append("")
    lines.append("### Per-arch coverage and mean over all tasks present")
    lines.append("")
    lines.append("| arch | n_tasks | mean tok_acc |")
    lines.append("|---|---:|---:|")
    for a in archs_10m:
        n, m = agg_10m[a]
        lines.append(f"| {a} | {n} | {fmt(m)} |")
    lines.append("")
    lines.append(
        f"### Shared-task ({len(shared_10m)} tasks, headwise_mamba2/headwise_stu/stu/attn all present)"
    )
    lines.append("")
    lines.append("| arch | mean tok_acc on shared |")
    lines.append("|---|---:|")
    for a in archs_10m:
        lines.append(f"| {a} | {fmt(means_10m_shared[a])} |")
    lines.append("")
    lines.append("### Wins of `headwise_stu` at 10m")
    lines.append("")
    lines.append("| opponent | wins / ties / n | win rate |")
    lines.append("|---|---|---:|")
    for label, t in [
        ("headwise_mamba2", w10_vs_m2),
        ("pure stu", w10_vs_pureSTU),
        ("attn", w10_vs_attn),
    ]:
        w, tied, n = t
        rate = (w / n) if n else 0.0
        lines.append(f"| {label} | {w} / {tied} / {n} | {rate:.0%} |")
    lines.append("")

    # section 6: takeaway
    lines.append("## 6. Takeaway")
    lines.append("")

    # build takeaway from numbers
    takeaway_bullets = []

    if d_hwstu_m1 is not None:
        if d_hwstu_m1 >= 0.02:
            takeaway_bullets.append(
                f"**STU swap helps the headwise hybrid vs Mamba-1** (Δmean {fmt_delta(d_hwstu_m1)}). "
                f"Some of the damage from the Mamba branch in the head-split is kernel-specific, not structural."
            )
        elif d_hwstu_m1 <= -0.02:
            takeaway_bullets.append(
                f"**STU does not rescue the headwise split.** Swapping Mamba-1 for STU in the head split "
                f"moves mean tok_acc {fmt_delta(d_hwstu_m1)} on the shared task set — the headwise architecture "
                f"is the problem, not the choice of SSM kernel."
            )
        else:
            takeaway_bullets.append(
                f"**STU is roughly a wash vs Mamba-1 inside the headwise split** (Δmean {fmt_delta(d_hwstu_m1)}). "
                f"Kernel choice is not the dominant factor; the head-split shape is."
            )

    if d_hwstu_pureSTU is not None:
        if d_hwstu_pureSTU >= 0.02:
            takeaway_bullets.append(
                f"Pairing STU with attention in a head-split **improves on pure STU** (Δmean "
                f"{fmt_delta(d_hwstu_pureSTU)}) — the hybrid is doing real work."
            )
        elif d_hwstu_pureSTU <= -0.02:
            takeaway_bullets.append(
                f"**Pure STU beats attn+STU headwise** (Δmean {fmt_delta(d_hwstu_pureSTU)} in favor of pure STU). "
                f"Splitting heads between attention and STU hurts more than attention adds — strong evidence "
                f"that the headwise split itself is the wrong shape, regardless of branch choice."
            )
        else:
            takeaway_bullets.append(
                f"attn+STU headwise ≈ pure STU (Δmean {fmt_delta(d_hwstu_pureSTU)}). The split adds no value; "
                f"the attention half is not earning its heads."
            )

    if d_hwstu_m2 is not None:
        if d_hwstu_m2 >= 0.02:
            takeaway_bullets.append(
                f"STU is a better branch than Mamba-2 inside the head-split "
                f"(Δmean {fmt_delta(d_hwstu_m2)} vs headwise_mamba2), so the Mamba-2 kernel was indeed a bad "
                f"fit for the split — but see the pure-STU comparison above for whether the split is worth it at all."
            )
        elif d_hwstu_m2 <= -0.02:
            takeaway_bullets.append(
                f"Even vs headwise_mamba2, STU does worse (Δmean {fmt_delta(d_hwstu_m2)}). "
                f"Both non-Mamba-1 kernels lose to the Mamba-1 baseline of the same split; the original "
                f"`headwise` configuration was not just lucky."
            )
        else:
            takeaway_bullets.append(
                f"STU ≈ Mamba-2 inside the head-split (Δmean {fmt_delta(d_hwstu_m2)})."
            )

    for b in takeaway_bullets:
        lines.append(f"- {b}")
    lines.append("")

    # one-line bottom line
    if d_hwstu_pureSTU is not None and d_hwstu_pureSTU <= 0 and d_hwstu_m1 is not None and d_hwstu_m1 <= 0:
        bottom = (
            "**Bottom line:** the headwise split is the wrong shape. Neither Mamba-2 nor STU saves it; "
            "pure STU beats attn+STU-headwise, and attn+Mamba-1 still beats attn+STU. The result from the "
            "Mamba-2 sweep generalizes — it's not about which branch kernel you pick."
        )
    elif d_hwstu_m1 is not None and d_hwstu_m1 > 0 and d_hwstu_pureSTU is not None and d_hwstu_pureSTU > 0:
        bottom = (
            "**Bottom line:** STU is a better branch than Mamba for the headwise split, and the hybrid "
            "genuinely beats pure STU. The kernel choice was the issue, not the split shape."
        )
    else:
        bottom = (
            "**Bottom line:** mixed — see the per-comparison bullets above. The headwise split is not "
            "cleanly rescued by STU, but it is also not uniformly harmful."
        )
    lines.append(bottom)
    lines.append("")

    # appendix
    lines.append("## 7. Notes")
    lines.append("")
    lines.append("### Anomalies / collapsed runs")
    lines.append("")
    if not anomalies:
        lines.append("No NaN or collapsed (`tok_acc ≤ 0.01`) runs detected across the four stores.")
    else:
        lines.append("| store | task | arch | scale | issue |")
        lines.append("|---|---|---|---|---|")
        for name, key, msg in anomalies:
            t, a, s = key
            lines.append(f"| {name} | {t} | {a} | {s} | {msg} |")
    lines.append("")
    lines.append(f"CSVs: `{OUT_1M}`, `{OUT_10M}`.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_1M} ({len(rows_1m)} rows)")
    print(f"wrote {OUT_10M} ({len(rows_10m)} rows)")

    # stdout quick debug
    print(f"\n1m shared tasks: {shared_n}")
    print(f"means (shared): headwise={fmt(means_1m_shared['headwise'])} "
          f"headwise_mamba2={fmt(means_1m_shared['headwise_mamba2'])} "
          f"headwise_stu={fmt(means_1m_shared['headwise_stu'])} "
          f"stu={fmt(means_1m_shared['stu'])}")
    print(f"d_hwstu_vs_m1={fmt_delta(d_hwstu_m1)} d_hwstu_vs_m2={fmt_delta(d_hwstu_m2)} "
          f"d_hwstu_vs_pureSTU={fmt_delta(d_hwstu_pureSTU)}")
    print(f"10m shared tasks: {len(shared_10m)}")
    if shared_10m:
        print(f"means (shared): headwise_mamba2={fmt(means_10m_shared['headwise_mamba2'])} "
              f"headwise_stu={fmt(means_10m_shared['headwise_stu'])} "
              f"stu={fmt(means_10m_shared['stu'])} attn={fmt(means_10m_shared['attn'])}")


if __name__ == "__main__":
    main()
