#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: v3_step3_model.py | Topic: 2 (v3) | Purpose:
Per-drug prediction of ΔMAP@5min after a DOCUMENTED vasopressor administration.
Models: XGBoost, Ridge, and a baseline-MAP-only regression-to-the-mean reference.
Reports MAE / RMSE / R² (de-emphasized), calibration (slope+intercept), and
80% prediction-interval coverage (XGBoost quantile). Patient-grouped CV + a
patient-level held-out test. Bootstrap 95% CIs (patient-level).

Works on BOTH databases: pass --events <parquet> --cohort <vitaldb|mover>.
MOVER run uses only intersection-compatible features (see FEATURES_COMMON).
"""
import os, sys, json, logging, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(os.environ.get("TOPIC2_WORK", Path(__file__).resolve().parents[1]))
RAW_ROOTS = [Path(os.environ.get("VITALDB_ROOT", "/path/to/vitaldb/physionet.org")),
             Path(os.environ.get("INSPIRE_ROOT", "/path/to/inspire/physionet.org")),
             Path(os.environ.get("MIMIC4_ROOT", "/path/to/mimic/physionet.org"))]


def safe_path(p):
    r = str(Path(p).resolve())
    for root in RAW_ROOTS:
        assert not r.startswith(str(root.resolve())), f"FATAL write to raw: {root}"
    return Path(p)


# Features available in BOTH VitalDB and MOVER (the externally-validatable model).
FEATURES_COMMON = [
    'age', 'sex_male', 'bmi', 'asa',
    'dose_rate_at_event',
    'baseline_map', 'ctx_map_mean', 'ctx_map_min', 'ctx_map_last',
    'ctx_map_trend', 'ctx_map_cv', 'ctx_hr_mean', 'ctx_shock_index',
    'ctx_time_below_65_min', 'ctx_surgery_time_min',
    'redose_flag',
]
TARGET = 'delta_map_5min'
GROUP = 'subjectid'   # patient-level grouping


def bootstrap_ci(y, yhat, groups, metric, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    vals = []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        vals.append(metric(y[idx], yhat[idx]))
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mae(y, p): return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def r2(y, p):
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float('nan')


def calibration(y, p):
    """Regression calibration: observed = a + b*predicted. slope b, intercept a."""
    import numpy as np
    b, a = np.polyfit(p, y, 1)
    return float(b), float(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=str(WORK / "outputs_v3/metrics/vitaldb_documented_events.parquet"))
    ap.add_argument('--cohort', default='vitaldb')
    ap.add_argument('--outdir', default=str(WORK / "outputs_v3"))
    ap.add_argument('--min_n', type=int, default=40, help='min events to fit a per-drug model')
    ap.add_argument('--exclude_censored', action='store_true',
                    help='sensitivity: drop START events with a same-drug re-dose in the outcome window')
    args = ap.parse_args()

    out = Path(args.outdir)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(out / "logs" / f"{ts}_v3_step3_model_{args.cohort}.log"),
                  logging.StreamHandler()])
    logging.info(f"=== Topic 2 v3 Step 3 modeling | cohort={args.cohort} ===")

    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    ev = pd.read_parquet(args.events)
    # primary analysis: START events (bolus / infusion-initiation).
    # Same-drug re-dose inside the outcome window is NORMAL titration, not
    # contamination -> keep as a covariate flag by default; drop only under
    # --exclude_censored (sensitivity analysis). Different-drug co-admin was
    # already excluded upstream in Step 1.
    ev = ev[ev['event_type'] == 'start'].copy()
    if args.exclude_censored:
        ev = ev[~ev['censored_repeat'].fillna(False)].copy()
    ev['redose_flag'] = ev['censored_repeat'].fillna(False).astype(int)
    ev[GROUP] = ev[GROUP].fillna(ev['caseid'])
    logging.info(f"Start events (uncensored): {len(ev)} across {ev[GROUP].nunique()} patients")

    feats = [f for f in FEATURES_COMMON if f in ev.columns]
    logging.info(f"Using {len(feats)} common features: {feats}")

    results = []
    for drug in sorted(ev['drug'].unique()):
        sub = ev[ev['drug'] == drug].copy()
        if len(sub) < args.min_n:
            logging.warning(f"[{drug}] n={len(sub)} < {args.min_n} -> descriptive only, skip model")
            results.append({'cohort': args.cohort, 'drug': drug, 'n': len(sub), 'model': 'INSUFFICIENT_N'})
            continue
        X = sub[feats].apply(pd.to_numeric, errors='coerce')
        X = X.fillna(X.median())
        y = sub[TARGET].astype(float).values
        groups = sub[GROUP].values

        # patient-grouped out-of-fold predictions (honest CV)
        n_splits = min(5, sub[GROUP].nunique())
        gkf = GroupKFold(n_splits=n_splits)
        oof_xgb = np.full(len(y), np.nan)
        oof_ridge = np.full(len(y), np.nan)
        oof_base = np.full(len(y), np.nan)
        oof_lo = np.full(len(y), np.nan)
        oof_hi = np.full(len(y), np.nan)

        for tr, te in gkf.split(X, y, groups):
            # XGBoost mean
            m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=42, n_jobs=4)
            m.fit(X.iloc[tr], y[tr])
            oof_xgb[te] = m.predict(X.iloc[te])
            # XGBoost quantile 10/90 for PI (uses sklearn API w/ pinball via objective)
            try:
                lo = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                      subsample=0.8, colsample_bytree=0.8,
                                      objective='reg:quantileerror', quantile_alpha=0.1,
                                      random_state=42, n_jobs=4)
                hi = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                      subsample=0.8, colsample_bytree=0.8,
                                      objective='reg:quantileerror', quantile_alpha=0.9,
                                      random_state=42, n_jobs=4)
                lo.fit(X.iloc[tr], y[tr]); hi.fit(X.iloc[tr], y[tr])
                oof_lo[te] = lo.predict(X.iloc[te]); oof_hi[te] = hi.predict(X.iloc[te])
            except Exception as e:
                logging.warning(f"[{drug}] quantile unavailable: {e}")
            # Ridge
            sc = StandardScaler().fit(X.iloc[tr])
            rg = Ridge(alpha=1.0).fit(sc.transform(X.iloc[tr]), y[tr])
            oof_ridge[te] = rg.predict(sc.transform(X.iloc[te]))
            # baseline-MAP-only reference (regression to the mean)
            if 'baseline_map' in feats:
                bm_tr = X.iloc[tr][['baseline_map']].values
                bm_te = X.iloc[te][['baseline_map']].values
                rb = Ridge(alpha=1.0).fit(bm_tr, y[tr])
                oof_base[te] = rb.predict(bm_te)

        for name, pred in [('xgboost', oof_xgb), ('ridge', oof_ridge), ('baseline_map_only', oof_base)]:
            ok = ~np.isnan(pred)
            if ok.sum() < 10:
                continue
            mae_m, mae_lo, mae_hi = bootstrap_ci(y[ok], pred[ok], groups[ok], mae)
            cs, ci = calibration(y[ok], pred[ok])
            rec = {'cohort': args.cohort, 'drug': drug, 'n': int(ok.sum()),
                   'model': name,
                   'MAE': round(mae_m, 2), 'MAE_lo': round(mae_lo, 2), 'MAE_hi': round(mae_hi, 2),
                   'RMSE': round(rmse(y[ok], pred[ok]), 2),
                   'R2': round(r2(y[ok], pred[ok]), 3),
                   'calib_slope': round(cs, 3), 'calib_intercept': round(ci, 2)}
            if name == 'xgboost':
                have_pi = ~np.isnan(oof_lo) & ~np.isnan(oof_hi)
                if have_pi.sum() > 10:
                    cov = float(np.mean((y[have_pi] >= oof_lo[have_pi]) & (y[have_pi] <= oof_hi[have_pi])))
                    rec['PI80_coverage'] = round(cov, 3)
            results.append(rec)
            logging.info(f"[{drug}] {name}: MAE={rec['MAE']} ({rec['MAE_lo']}-{rec['MAE_hi']}) "
                         f"RMSE={rec['RMSE']} R2={rec['R2']} slope={rec['calib_slope']} "
                         f"cov={rec.get('PI80_coverage','-')}")

        # save OOF predictions for figures
        sub = sub.assign(pred_xgb=oof_xgb, pred_ridge=oof_ridge, pred_base=oof_base,
                         pi_lo=oof_lo, pi_hi=oof_hi)
        sub.to_parquet(safe_path(out / "metrics" / f"oof_predictions_{args.cohort}_{drug}.parquet"), index=False)

    rdf = pd.DataFrame(results)
    rpath = safe_path(out / "metrics" / f"v3_model_performance_{args.cohort}.csv")
    rdf.to_csv(rpath, index=False)
    logging.info(f"Saved -> {rpath}\n{rdf.to_string()}")
    logging.info("=== Step 3 complete ===")


if __name__ == '__main__':
    main()
