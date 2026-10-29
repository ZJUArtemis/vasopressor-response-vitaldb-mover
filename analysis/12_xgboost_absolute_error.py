#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Refit XGBoost with an absolute-error objective on the identical patient folds, so that
the gradient-boosted and linear models can be compared at matched training objective.

The squared-error XGBoost in the primary analysis and the median regression differ in both
model class and training objective. This script fits XGBRegressor(objective='reg:absoluteerror')
with the same randomised search budget scored on mean absolute error, for each primary drug
and both feature sets, and reports the paired dMAE against the squared-error XGBoost, the
median regression, tuned Ridge and the baseline-MAP reference.

Inputs:  data/event_master.parquet; folds and out-of-fold arrays from 01, 10
Outputs (results/): mae_xgb_comparison.csv, mae_xgb_delta.csv, mae_xgb_oof_<feat>_<drug>.npy
"""
import logging
import importlib.util
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
V14 = RES; V19 = RES; OUT = RES
_s = importlib.util.spec_from_file_location("mv", ROOT / "analysis" / "01_fit_regression_models.py")
mv = importlib.util.module_from_spec(_s); _s.loader.exec_module(mv)
_s2 = importlib.util.spec_from_file_location("m19", ROOT / "analysis" / "10_fair_linear_comparators.py")
m19 = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(m19)

DRUGS = ["phenylephrine", "ephedrine", "vasopressin"]
SEED = mv.SEED; N_BOOT = 2000; N_JOBS = 4
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(OUT / f"xgb_absolute_error_{datetime.now():%Y%m%d_%H%M%S}.log"),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)


def space():
    return {"m__n_estimators": randint(100, 800), "m__max_depth": randint(3, 8),
            "m__learning_rate": uniform(0.01, 0.19), "m__subsample": uniform(0.6, 0.4),
            "m__colsample_bytree": uniform(0.6, 0.4), "m__reg_lambda": uniform(0.1, 4.9)}


def _fold(f, y, g, te, num_cols):
    tr = ~te
    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler())]), num_cols),
         ("unit", mv.UnitDummies(min_count=20), "dose_unit")], remainder="drop")
    pipe = Pipeline([("pre", pre), ("m", XGBRegressor(objective="reg:absoluteerror",
                                                      random_state=SEED, n_jobs=2))])
    inner = GroupKFold(n_splits=min(3, len(np.unique(g[tr]))))
    s = RandomizedSearchCV(pipe, space(), n_iter=12, scoring="neg_mean_absolute_error",
                           cv=inner.split(f.iloc[tr], y[tr], g[tr]), random_state=SEED, n_jobs=1)
    s.fit(f.iloc[tr], y[tr], groups=g[tr])
    return te, s.best_estimator_.predict(f.iloc[te])


def main():
    d = mv.load()
    delta_rows, comp_rows = [], []
    for drug in DRUGS:
        sub = d[(d.drug == drug) & (d.event_class_v12 == "iv_bolus")].copy()
        y = sub["delta_map_5min"].values.astype(float); g = sub["subjectid"].values
        folds = m19.load_folds(drug)
        num_base = [c for c in mv.BASE_NUM if c in sub.columns]
        num_hist = num_base + [c for c in mv.HIST if c in sub.columns]
        log.info(f"=== {drug} n={len(sub)} ===")
        store = {}
        for feat, hist, nc in [("base", False, num_base), ("base+hist", True, num_hist)]:
            fl = feat.replace("+", "_")
            cached = OUT / f"mae_xgb_oof_{fl}_{drug}.npy"
            if cached.exists():
                oof = np.load(cached)
                log.info(f"[{drug}] {feat} xgb_mae loaded from saved OOF array")
            else:
                Xnum, unit = mv.build_matrix(sub, hist)
                f = Xnum.copy(); f["dose_unit"] = unit.values
                jobs = [(r, np.isin(g, tp)) for r, fold in enumerate(folds) for tp in fold]
                res = Parallel(n_jobs=N_JOBS)(delayed(_fold)(f, y, g, te, nc) for _, te in jobs)
                oof = np.full((len(folds), len(y)), np.nan)
                for (r, _), (te, pred) in zip(jobs, res):
                    oof[r, te] = pred
                np.save(cached, oof)
            store[(feat, "xgb_mae")] = oof
            pp = [float(np.mean(np.abs(y - oof[r]))) for r in range(oof.shape[0])]
            comp_rows.append({"drug": drug, "features": feat, "model": "xgb_mae",
                              "mae_mean": round(float(np.mean(pp)), 3),
                              "mae_min": round(min(pp), 3), "mae_max": round(max(pp), 3),
                              "n": int(len(sub)), "n_patients": int(len(np.unique(g)))})
            log.info(f"[{drug}] {feat} xgb_mae MAE={np.mean(pp):.3f}")
        # comparators on the same folds
        for feat in ["base", "base+hist"]:
            fl = feat.replace("+", "_")
            store[(feat, "xgb_sq")] = np.load(V14 / f"oof_{fl}_xgb_{drug}.npy")
            store[(feat, "median")] = np.load(V19 / f"oof_{fl}_median_{drug}.npy")
            store[(feat, "ridge_tuned")] = np.load(V19 / f"oof_{fl}_ridge_tuned_{drug}.npy")
        store[("base", "baseline")] = np.load(V14 / f"oof_baseline_ridge_{drug}.npy")
        comps = [("base", "xgb_mae", "base", "xgb_sq", "mae_xgb_vs_squared_xgb_base"),
                 ("base+hist", "xgb_mae", "base+hist", "xgb_sq", "mae_xgb_vs_squared_xgb_hist"),
                 ("base", "xgb_mae", "base", "median", "mae_xgb_vs_median_base"),
                 ("base+hist", "xgb_mae", "base+hist", "median", "mae_xgb_vs_median_hist"),
                 ("base", "xgb_mae", "base", "ridge_tuned", "mae_xgb_vs_ridge_tuned_base"),
                 ("base+hist", "xgb_mae", "base", "baseline", "mae_xgb_hist_vs_baseline")]
        for fa, ka, fb, kb, label in comps:
            A = store[(fa, ka)]; B = store[(fb, kb)]
            dd = np.abs(y - A) - np.abs(y - B)
            lo, hi = m19.cluster_bootstrap(dd, g)
            per = dd.mean(axis=1)
            delta_rows.append({"drug": drug, "comparison": label, "A": f"{ka}:{fa}", "B": f"{kb}:{fb}",
                               "n": int(len(sub)), "n_patients": int(len(np.unique(g))),
                               "dmae": round(float(dd.mean()), 3), "ci_lo": round(lo, 3),
                               "ci_hi": round(hi, 3), "sd_across_partitions": round(float(per.std(ddof=1)), 3),
                               "min_partition": round(float(per.min()), 3),
                               "max_partition": round(float(per.max()), 3)})
            log.info(f"[{drug}] {label}: {dd.mean():+.3f} ({lo:+.3f},{hi:+.3f})")
    pd.DataFrame(comp_rows).to_csv(OUT / "mae_xgb_comparison.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(OUT / "mae_xgb_delta.csv", index=False)
    log.info("absolute-error XGBoost done")


if __name__ == "__main__":
    main()
