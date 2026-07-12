"""Drop-primitive ablation against 50M LM CE.

For each primitive P, compute mech-suite mean accuracy excluding P's
tasks, correlate with c4_en CE across the held-out architectures, and
compare to the full-suite baseline.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from primitive_taxonomy import PRIMARY_PRIMITIVE, PRIMITIVES, FAMILY_OF_PRIMITIVE, FAMILIES

ROOT = Path(__file__).resolve().parent
WIDE = pd.read_csv(ROOT / "unified_1m.csv").set_index("task")
LM = pd.read_csv(ROOT / "lm_ce.csv")
LM50 = LM[LM.scale == "50m"].set_index("arch")
LM150 = LM[LM.scale == "150m"].set_index("arch")

# Drop archs with <50% coverage for the per-task table
coverage = (WIDE.notna().sum(axis=0) / len(WIDE.index))
print("arch coverage:")
print(coverage.to_string())
keep_archs = coverage[coverage > 0.5].index.tolist()
print(f"keep_archs ({len(keep_archs)}):", keep_archs)
WIDE = WIDE[keep_archs]

# Drop tasks with any NaN among kept archs (for clean statistics)
keep_tasks = WIDE.notna().all(axis=1)
WIDE = WIDE[keep_tasks]
print(f"complete-coverage tasks: {WIDE.shape[0]}")

# Map tasks to primitives/families
prim_of = WIDE.index.map(PRIMARY_PRIMITIVE)
fam_of = WIDE.index.map(lambda t: FAMILY_OF_PRIMITIVE.get(PRIMARY_PRIMITIVE.get(t, ""), "?"))
print("\nprimitive coverage in usable tasks:")
print(pd.Series(prim_of).value_counts().to_string())

def _score(df_subset: pd.DataFrame, lm_index: pd.DataFrame, archs: list[str]) -> tuple:
    if df_subset.empty:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    suite_mean = df_subset[archs].mean(axis=0)
    ce = lm_index.loc[archs, "c4_en_CE"]
    rho, prho = spearmanr(suite_mean, ce)
    r, pr = pearsonr(suite_mean, ce)
    return (rho, prho, r, pr, df_subset.shape[0])

def ablate(scale: str, lm_index: pd.DataFrame) -> pd.DataFrame:
    archs = sorted(set(WIDE.columns) & set(lm_index.index))
    print(f"\n[{scale}] shared archs ({len(archs)}):", archs)
    rows = []
    rho, prho, r, pr, n = _score(WIDE, lm_index, archs)
    rows.append({"drop": "(none)", "level": "baseline", "n_tasks": n, "n_archs": len(archs),
                 "rho": rho, "rho_p": prho, "pearson": r, "pearson_p": pr})
    for p in sorted(set(prim_of)):
        sub = WIDE[prim_of != p]
        rho, prho, r, pr, n = _score(sub, lm_index, archs)
        rows.append({"drop": p, "level": "primitive", "n_tasks": n, "n_archs": len(archs),
                     "rho": rho, "rho_p": prho, "pearson": r, "pearson_p": pr})
    for f in sorted(set(fam_of)):
        sub = WIDE[fam_of != f]
        rho, prho, r, pr, n = _score(sub, lm_index, archs)
        rows.append({"drop": f"family:{f}", "level": "family", "n_tasks": n, "n_archs": len(archs),
                     "rho": rho, "rho_p": prho, "pearson": r, "pearson_p": pr})
    return pd.DataFrame(rows).assign(scale=scale)

abl = pd.concat([ablate("50m", LM50), ablate("150m", LM150)], ignore_index=True)
abl.to_csv(ROOT / "drop_family_ablation.csv", index=False)
print("\n--- Drop-family ablation ---")
print(abl.to_string(index=False))
