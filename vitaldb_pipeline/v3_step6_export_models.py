#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: v3_step6_export_models.py | Topic: 2 (v3) | Purpose:
Train final per-drug models on the FULL VitalDB documented cohort and export them
as pickled bundles for external validation on MOVER. Only intersection-compatible
features are used so the bundle applies cleanly to MOVER.
Bundles are copied into mover_package/models/vitaldb/<drug>.pkl.
"""
import os, json, logging, pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(os.environ.get("TOPIC2_WORK", Path(__file__).resolve().parents[1]))
OUT = WORK / "outputs_v3"
PKG_MODELS = WORK / "mover_package" / "models" / "vitaldb"
RAW_ROOTS = [Path(os.environ.get("VITALDB_ROOT", "/path/to/vitaldb/physionet.org")),
             Path(os.environ.get("INSPIRE_ROOT", "/path/to/inspire/physionet.org")),
             Path(os.environ.get("MIMIC4_ROOT", "/path/to/mimic/physionet.org"))]


def safe_path(p):
    r = str(Path(p).resolve())
    for root in RAW_ROOTS:
        assert not r.startswith(str(root.resolve())), f"FATAL write to raw: {root}"
    return Path(p)


# MOVER lacks HR/shock_index + anesthetic rates -> intersection excludes them
FEATURES_COMMON = ['age', 'sex_male', 'bmi', 'asa', 'dose_rate_at_event',
                   'baseline_map', 'ctx_map_mean', 'ctx_map_min', 'ctx_map_last',
                   'ctx_map_trend', 'ctx_map_cv', 'ctx_time_below_65_min',
                   'ctx_surgery_time_min', 'redose_flag']
TARGET = 'delta_map_5min'


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(OUT / "logs" / f"{ts}_v3_step6_export.log"), logging.StreamHandler()])
    import xgboost as xgb
    PKG_MODELS.mkdir(parents=True, exist_ok=True)

    ev = pd.read_parquet(OUT / "metrics" / "vitaldb_documented_events.parquet")
    ev = ev[ev['event_type'] == 'start'].copy()
    ev['redose_flag'] = ev['censored_repeat'].fillna(False).astype(int)
    feats = [f for f in FEATURES_COMMON if f in ev.columns]

    for drug in sorted(ev['drug'].unique()):
        sub = ev[ev['drug'] == drug]
        if len(sub) < 40:
            logging.warning(f"[{drug}] n={len(sub)} < 40; export Ridge fallback only")
        X = sub[feats].apply(pd.to_numeric, errors='coerce'); med = X.median()
        X = X.fillna(med)
        y = sub[TARGET].astype(float).values
        model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=42, n_jobs=4).fit(X, y)
        bundle = {'drug': drug, 'features': feats, 'medians': med.to_dict(),
                  'model': model, 'trained_on': 'VitalDB documented cohort',
                  'n': int(len(sub)), 'created': ts}
        p = safe_path(PKG_MODELS / f"{drug}.pkl")
        pickle.dump(bundle, open(p, 'wb'))
        logging.info(f"[{drug}] exported bundle (n={len(sub)}) -> {p}")
    logging.info("=== Step 6 complete ===")


if __name__ == '__main__':
    main()
