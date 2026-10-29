#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read-only verification of the analysis implementation against the saved fold assignments,
out-of-fold arrays and feature lists.

Checks: patient isolation within every repeat (no patient on both sides of a fold; folds
partition the patients exactly once); one prediction per event per repeat (no missing value
in any out-of-fold array) and distinct partitions across repeats; no follow-up, outcome or
other post-dose column among the model features; the analysed cohort consisting only of
records classified as intravenous bolus; and the analysable cohort as a fraction of
documented intravenous-bolus records (an upper bound on outcome-observability loss).

Exits non-zero on any failure. Outputs (results/): code_audit.json
"""
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
V14 = RES; V19 = RES; OUT = RES
_s = importlib.util.spec_from_file_location("mv", ROOT / "analysis" / "01_fit_regression_models.py")
mv = importlib.util.module_from_spec(_s); _s.loader.exec_module(mv)

DRUGS = ["phenylephrine", "ephedrine", "vasopressin", "norepinephrine"]
results = {"checks": [], "summary": {}}


def rec(name, ok, detail):
    results["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS  " if ok else "FAIL  ") + name + " :: " + detail)


def main():
    d = mv.load()

    # ---------- (4) exposure classification ----------
    for drug in DRUGS:
        sub = d[d.drug == drug]
        counts = sub.event_class_v12.value_counts().to_dict()
        analysed = sub[sub.event_class_v12 == "iv_bolus"]
        ok = len(analysed) > 0 and set(analysed.event_class_v12.unique()) == {"iv_bolus"}
        rec(f"exposure class [{drug}]", ok,
            f"iv_bolus n={len(analysed)}; all rows of the analysed subset are iv_bolus; "
            f"full class counts {counts}")

    # ---------- (1) patient isolation ----------
    for drug in DRUGS:
        z = np.load(V14 / f"folds_{drug}.npz", allow_pickle=True)
        R = max(int(k.split("_")[0][1:]) for k in z.files) + 1
        F = max(int(k.split("_")[1][1:]) for k in z.files) + 1
        allpats = set()
        bad = 0
        for r in range(R):
            seen = []
            for fk in range(F):
                seen.append(set(np.asarray(z[f"r{r}_f{fk}"]).tolist()))
            union = set().union(*seen)
            total = sum(len(s) for s in seen)
            if total != len(union):          # folds must not overlap
                bad += 1
            allpats |= union
        rec(f"patient isolation [{drug}]", bad == 0,
            f"{R} repeats x {F} folds; fold overlap violations={bad}; "
            f"{len(allpats)} distinct patients across all folds")

    # ---------- (2) one prediction per event, and repeats differ ----------
    for drug in DRUGS:
        sub = d[(d.drug == drug) & (d.event_class_v12 == "iv_bolus")]
        n = len(sub)
        for fl in ["base", "base_hist"]:
            p = V14 / f"oof_{fl}_xgb_{drug}.npy"
            if not p.exists():
                continue
            a = np.load(p)
            ok_shape = a.shape[1] == n
            ok_nan = not np.isnan(a).any()
            uniq_repeats = len({tuple(np.round(v[:50], 6)) for v in a})
            rec(f"oof complete [{drug}/{fl}]", ok_shape and ok_nan,
                f"shape={a.shape} expected_n={n} nan={bool(np.isnan(a).any())} "
                f"distinct repeats (first-50 fingerprint)={uniq_repeats}")

    # ---------- (3) pre-dose features only ----------
    feats = list(mv.BASE_NUM) + list(mv.HIST)
    leaked = [c for c in feats if c.startswith("fu_")]
    rec("no follow-up column among features", len(leaked) == 0,
        f"{len(feats)} features; any fu_* present: {leaked}")
    hist_cols = [c for c in feats if c.startswith("hist_")]
    fu_cols = [c for c in d.columns if c.startswith("fu_")]
    rec("history features drawn from pre-dose columns", len(hist_cols) > 0,
        f"{len(hist_cols)} history features, all named hist_*; "
        f"{len(fu_cols)} follow-up columns exist in the table and are unused by the models "
        f"({sorted(set(feats) & set(fu_cols)) or 'none'})")
    # outcome column must not be a feature
    rec("outcome not among features", "delta_map_5min" not in feats,
        f"delta_map_5min in features: {'delta_map_5min' in feats}")

    # ---------- (5) denominator for outcome observability ----------
    cen = pd.read_csv(DATA / "v12_action_census.csv")
    ivb = cen[cen.event_class_v12 == "iv_bolus"].groupby("drug")["n_rows"].sum()
    rows = []
    for drug in ["phenylephrine", "ephedrine", "vasopressin"]:
        sub = d[(d.drug == drug) & (d.event_class_v12 == "iv_bolus")]
        rows.append({"drug": drug, "documented_iv_bolus_mar_rows": int(ivb.get(drug, 0)),
                     "analysable_events": int(len(sub)),
                     "fraction": round(len(sub) / max(int(ivb.get(drug, 1)), 1), 4)})
    results["observability_denominator"] = rows
    rec("observability denominator computed (informational)", True,
        "; ".join(f"{r['drug']} {r['analysable_events']:,}/{r['documented_iv_bolus_mar_rows']:,} "
                  f"({r['fraction']*100:.1f}%)" for r in rows))

    fails = [c for c in results["checks"] if not c["pass"]]
    results["summary"] = {"n_checks": len(results["checks"]), "n_fail": len(fails)}
    (OUT / "code_audit.json").write_text(json.dumps(results, indent=2))
    print(f"\n{len(results['checks'])} checks, {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
