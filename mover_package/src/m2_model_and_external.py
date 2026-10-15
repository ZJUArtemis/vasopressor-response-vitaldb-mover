#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
m2_model_and_external.py | Topic 2 v3 | MOVER server.

Three analyses, per drug (phenylephrine, ephedrine, norepinephrine, ...):
  (A) MOVER-native model: XGBoost / Ridge / baseline-MAP-only, patient-grouped OOF,
      bootstrap 95% CI, 80% PI coverage. (The headline results — big N.)
  (B) External validation of the VitalDB-trained model on MOVER (if a VitalDB model
      bundle is shipped in models/vitaldb/). Reports MAE + calibration BEFORE and
      AFTER recalibration (linear + isotonic).
  (C) Transportability summary: per-drug external MAE/calibration table + subgroup.

Features restricted to the VitalDB/MOVER intersection (FEATURES_COMMON). CPU only.
Output: outputs/m2_mover_native.csv, outputs/m2_external_validation.csv, OOF parquets.
"""
import os, json, logging, argparse, pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "outputs"

# intersection features; MOVER lacks HR/shock_index + propofol/remi -> excluded here
FEATURES_COMMON = ['age', 'sex_male', 'bmi', 'asa', 'dose_rate_at_event',
                   'baseline_map', 'ctx_map_mean', 'ctx_map_min', 'ctx_map_last',
                   'ctx_map_trend', 'ctx_map_cv', 'ctx_time_below_65_min',
                   'ctx_surgery_time_min', 'redose_flag']
TARGET = 'delta_map_5min'
GROUP = 'subjectid'


def mae(y, p): return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def r2(y, p):
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float('nan')
def calib(y, p):
    b, a = np.polyfit(p, y, 1); return float(b), float(a)


def bootstrap_ci(y, yhat, groups, metric, n=1000, seed=42):
    rng = np.random.default_rng(seed); uniq = np.unique(groups); vals = []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        vals.append(metric(y[idx], yhat[idx]))
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=str(OUT / "mover_documented_events.parquet"))
    ap.add_argument('--vitaldb_models', default=str(PKG / "models" / "vitaldb"),
                    help='dir with per-drug VitalDB model bundles (optional, for external validation)')
    ap.add_argument('--min_n', type=int, default=100)
    args = ap.parse_args()
    OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(OUT / "logs" / f"{ts}_m2_model.log"), logging.StreamHandler()])
    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    ev = pd.read_parquet(args.events)
    ev = ev[ev['event_type'] == 'start'].copy()
    ev['redose_flag'] = ev['censored_repeat'].fillna(False).astype(int)
    ev[GROUP] = ev[GROUP].fillna(ev['caseid'])
    feats = [f for f in FEATURES_COMMON if f in ev.columns]
    logging.info(f"MOVER start events: {len(ev)} | features: {feats}")

    native, external = [], []
    for drug in sorted(ev['drug'].unique()):
        sub = ev[ev['drug'] == drug].copy()
        if len(sub) < args.min_n:
            logging.warning(f"[{drug}] n={len(sub)} < {args.min_n}; descriptive only")
            native.append({'drug': drug, 'n': len(sub), 'model': 'INSUFFICIENT_N'})
            continue
        X = sub[feats].apply(pd.to_numeric, errors='coerce'); X = X.fillna(X.median())
        y = sub[TARGET].astype(float).values; groups = sub[GROUP].values

        # (A) MOVER-native OOF
        gkf = GroupKFold(n_splits=min(5, sub[GROUP].nunique()))
        oof = {k: np.full(len(y), np.nan) for k in ['xgb', 'ridge', 'base', 'lo', 'hi']}
        for tr, te in gkf.split(X, y, groups):
            m = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=42, n_jobs=4).fit(X.iloc[tr], y[tr])
            oof['xgb'][te] = m.predict(X.iloc[te])
            try:
                for q, key in [(0.1, 'lo'), (0.9, 'hi')]:
                    qm = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03,
                                          objective='reg:quantileerror', quantile_alpha=q,
                                          random_state=42, n_jobs=4).fit(X.iloc[tr], y[tr])
                    oof[key][te] = qm.predict(X.iloc[te])
            except Exception as e:
                logging.warning(f"[{drug}] quantile NA: {e}")
            sc = StandardScaler().fit(X.iloc[tr])
            oof['ridge'][te] = Ridge(alpha=1.0).fit(sc.transform(X.iloc[tr]), y[tr]).predict(sc.transform(X.iloc[te]))
            oof['base'][te] = Ridge(alpha=1.0).fit(X.iloc[tr][['baseline_map']].values, y[tr]).predict(X.iloc[te][['baseline_map']].values)
        for name, key in [('xgboost', 'xgb'), ('ridge', 'ridge'), ('baseline_map_only', 'base')]:
            p = oof[key]; ok = ~np.isnan(p)
            m_, lo, hi = bootstrap_ci(y[ok], p[ok], groups[ok], mae)
            cs, ci = calib(y[ok], p[ok])
            rec = {'drug': drug, 'n': int(ok.sum()), 'model': name, 'MAE': round(m_, 2),
                   'MAE_lo': round(lo, 2), 'MAE_hi': round(hi, 2), 'RMSE': round(rmse(y[ok], p[ok]), 2),
                   'R2': round(r2(y[ok], p[ok]), 3), 'calib_slope': round(cs, 3), 'calib_intercept': round(ci, 2)}
            if name == 'xgboost':
                pi = ~np.isnan(oof['lo']) & ~np.isnan(oof['hi'])
                if pi.sum() > 10:
                    rec['PI80_coverage'] = round(float(np.mean((y[pi] >= oof['lo'][pi]) & (y[pi] <= oof['hi'][pi]))), 3)
            native.append(rec)
            logging.info(f"[{drug}] native {name}: MAE={rec['MAE']} slope={rec['calib_slope']}")
        sub.assign(**{f'pred_{k}': oof[k] for k in oof}).to_parquet(OUT / f"oof_mover_{drug}.parquet", index=False)

        # (B) external validation: VitalDB-trained model applied to MOVER
        bundle = Path(args.vitaldb_models) / f"{drug}.pkl"
        if bundle.exists():
            vb = pickle.load(open(bundle, 'rb'))
            vfeats = vb['features']
            Xv = sub[[f for f in vfeats if f in sub.columns]].apply(pd.to_numeric, errors='coerce')
            for f in vfeats:
                if f not in Xv.columns:
                    Xv[f] = np.nan
            Xv = Xv[vfeats].fillna(pd.Series(vb.get('medians', {})))
            pred = vb['model'].predict(Xv)
            m0, lo0, hi0 = bootstrap_ci(y, pred, groups, mae)
            cs0, ci0 = calib(y, pred)
            # recalibration
            b, a = np.polyfit(pred, y, 1); pred_lin = a + b * pred
            iso = IsotonicRegression(out_of_bounds='clip').fit(pred, y); pred_iso = iso.predict(pred)
            external.append({'drug': drug, 'n': len(y), 'stage': 'before_recalib',
                             'MAE': round(m0, 2), 'MAE_lo': round(lo0, 2), 'MAE_hi': round(hi0, 2),
                             'calib_slope': round(cs0, 3), 'calib_intercept': round(ci0, 2)})
            external.append({'drug': drug, 'n': len(y), 'stage': 'linear_recalib',
                             'MAE': round(mae(y, pred_lin), 2),
                             'calib_slope': round(calib(y, pred_lin)[0], 3)})
            external.append({'drug': drug, 'n': len(y), 'stage': 'isotonic_recalib',
                             'MAE': round(mae(y, pred_iso), 2)})
            logging.info(f"[{drug}] EXTERNAL VitalDB->MOVER: MAE {m0:.2f} slope {cs0:.2f} "
                         f"-> lin {mae(y,pred_lin):.2f} / iso {mae(y,pred_iso):.2f}")
        else:
            logging.info(f"[{drug}] no VitalDB bundle at {bundle} -> skip external validation")

    pd.DataFrame(native).to_csv(OUT / "m2_mover_native.csv", index=False)
    pd.DataFrame(external).to_csv(OUT / "m2_external_validation.csv", index=False)
    logging.info("saved m2_mover_native.csv + m2_external_validation.csv")
    logging.info("=== m2 complete ===")


if __name__ == '__main__':
    main()
