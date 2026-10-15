#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: v3_step7_iv_sensitivity.py | Topic: 2 (v3) | Purpose:
IV-restricted + dose-unit-stratified sensitivity analysis on the RETURNED MOVER
event table. The operator flagged that the main MOVER cohort mixes administration
routes (epinephrine START is majority non-IV local-anaesthetic adjuvant) and that
dose numbers are not dimensionally harmonised. This script re-runs the MOVER-native
per-drug models on the IntraVENOUS-only subset, reusing the operator's EXACT modeling
logic (local_model_and_external.py), so the two cohorts are directly comparable.

Reads : mover_returned/outputs/mover_documented_events.parquet  (read-only)
Writes: outputs_v3/metrics/m2_mover_native_IVonly.csv
        outputs_v3/metrics/iv_sensitivity_summary.json
No raw-data paths are touched.
"""
import os, json, logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(os.environ.get("TOPIC2_WORK", Path(__file__).resolve().parents[1]))
RET = WORK / "mover_returned" / "outputs" / "mover_documented_events.parquet"
OUT = WORK / "outputs_v3"
RAW_ROOTS = [Path(os.environ.get("VITALDB_ROOT", "/path/to/vitaldb/physionet.org")),
             Path(os.environ.get("INSPIRE_ROOT", "/path/to/inspire/physionet.org")),
             Path(os.environ.get("MIMIC4_ROOT", "/path/to/mimic/physionet.org"))]


def safe_path(p):
    r = str(Path(p).resolve())
    for root in RAW_ROOTS:
        assert not r.startswith(str(root.resolve())), f"FATAL write to raw: {root}"
    return Path(p)


# ---- operator's exact FEATURES_COMMON + metrics (local_model_and_external.py) ----
FEATURES_COMMON = ['age', 'sex_male', 'bmi', 'asa', 'dose_rate_at_event',
                   'baseline_map', 'ctx_map_mean', 'ctx_map_min', 'ctx_map_last',
                   'ctx_map_trend', 'ctx_map_cv', 'ctx_time_below_65_min',
                   'ctx_surgery_time_min']
TARGET = 'delta_map_5min'
GROUP = 'subjectid'


def mae(y, p): return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def r2(y, p):
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float('nan')
def calib(y, p):
    b, a = np.polyfit(p, y, 1); return float(b), float(a)


def bootstrap_ci(y, yhat, groups, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    totals = np.bincount(inverse, weights=np.abs(y - yhat))
    sizes = np.bincount(inverse)
    vals = []
    for _ in range(n):
        samp = rng.integers(0, len(uniq), size=len(uniq))
        vals.append(totals[samp].sum() / sizes[samp].sum())
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_native(ev, feats, min_n=100):
    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    rows = []
    for drug in sorted(ev['drug'].unique()):
        sub = ev[ev['drug'] == drug].copy()
        if len(sub) < min_n:
            rows.append({'drug': drug, 'n': len(sub), 'model': 'INSUFFICIENT_N'})
            continue
        X = sub[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        y = sub[TARGET].astype(float).values
        groups = sub[GROUP].values
        gkf = GroupKFold(n_splits=min(5, sub[GROUP].nunique()))
        oof = {k: np.full(len(y), np.nan) for k in ['xgb', 'ridge', 'base']}
        for tr, te in gkf.split(X, y, groups):
            medians = X.iloc[tr].median().fillna(0)
            Xtr, Xte = X.iloc[tr].fillna(medians), X.iloc[te].fillna(medians)
            m = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=42, n_jobs=4).fit(Xtr, y[tr])
            oof['xgb'][te] = m.predict(Xte)
            sc = StandardScaler().fit(Xtr)
            oof['ridge'][te] = Ridge(alpha=1.0).fit(sc.transform(Xtr), y[tr]).predict(sc.transform(Xte))
            oof['base'][te] = Ridge(alpha=1.0).fit(Xtr[['baseline_map']].values, y[tr]).predict(Xte[['baseline_map']].values)
        for name, key in [('xgboost', 'xgb'), ('ridge', 'ridge'), ('baseline_map_only', 'base')]:
            p = oof[key]; ok = ~np.isnan(p)
            m_, lo, hi = bootstrap_ci(y[ok], p[ok], groups[ok])
            cs, ci = calib(y[ok], p[ok])
            rows.append({'drug': drug, 'n': int(ok.sum()), 'model': name,
                         'MAE': round(mae(y[ok], p[ok]), 2), 'MAE_lo': round(lo, 2), 'MAE_hi': round(hi, 2),
                         'RMSE': round(rmse(y[ok], p[ok]), 2), 'R2': round(r2(y[ok], p[ok]), 3),
                         'calib_slope': round(cs, 3), 'calib_intercept': round(ci, 2)})
            logging.info(f"[{drug}][IVonly] {name}: MAE={rows[-1]['MAE']} slope={rows[-1]['calib_slope']}")
    return rows


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(OUT / "logs" / f"{ts}_v3_step7_iv_sensitivity.log"), logging.StreamHandler()])
    logging.info("=== Topic 2 v3 Step 7: IV-restricted sensitivity (reusing operator m2 logic) ===")

    ev = pd.read_parquet(RET)
    ev = ev[ev['event_type'] == 'start'].copy()
    ev[GROUP] = ev[GROUP].fillna(ev['caseid'])
    # dose-unit one-hot exactly as operator did
    units = pd.get_dummies(ev['dose_unit'].fillna('missing'), prefix='dose_unit', dtype=float)
    ev = pd.concat([ev, units], axis=1)
    feats = [f for f in FEATURES_COMMON if f in ev.columns] + list(units.columns)

    # route composition census
    s_all = ev.copy()
    iv = ev[ev['med_route'] == 'IntraVENOUS'].copy()
    census = {}
    for d in sorted(s_all['drug'].unique()):
        n_all = int((s_all['drug'] == d).sum())
        n_iv = int((iv['drug'] == d).sum())
        census[d] = {'all_route': n_all, 'IV_only': n_iv,
                     'nonIV_dropped': n_all - n_iv, 'IV_frac': round(n_iv / n_all, 3)}
    logging.info(f"route census: {json.dumps(census, indent=2)}")

    # re-fit units one-hot WITHIN the IV subset so columns are valid
    iv_units = pd.get_dummies(iv['dose_unit'].fillna('missing'), prefix='dose_unit', dtype=float)
    iv = pd.concat([iv.drop(columns=[c for c in iv.columns if c.startswith('dose_unit_')]), iv_units], axis=1)
    iv_feats = [f for f in FEATURES_COMMON if f in iv.columns] + list(iv_units.columns)

    rows = run_native(iv, iv_feats, min_n=100)
    df = pd.DataFrame(rows)
    outp = safe_path(OUT / "metrics" / "m2_mover_native_IVonly.csv")
    df.to_csv(outp, index=False)
    logging.info(f"saved {outp}")

    summ = {'generated': ts, 'route_census': census,
            'note': 'IV-only native models; dose numbers remain dimensionally mixed '
                    '(mcg bolus vs mcg/min infusion) within drug, same limitation as main analysis.',
            'IVonly_results': rows}
    with open(safe_path(OUT / "metrics" / "iv_sensitivity_summary.json"), 'w') as f:
        json.dump(summ, f, indent=2)
    logging.info("=== Step 7 complete ===")


if __name__ == '__main__':
    main()
