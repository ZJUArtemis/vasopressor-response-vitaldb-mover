#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Route audit.

The exposure keyword rule excluded recorded non-intravenous routes but did not include
"intramuscular", so seven analysed administrations carry that route. This script reports, per
drug and per model, the number of records by recorded route and the mean absolute error on the
full analysed set and on the explicitly intravenous subset.

This is a re-score of saved out-of-fold predictions on a restricted evaluation set. No model is
retrained, so each model's training cohort still contains the seven records.

Inputs:  data/event_master.parquet; out-of-fold arrays from 01, 10, 12
Outputs (results/): route_audit.csv
"""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; RES = ROOT / "results"
_s = importlib.util.spec_from_file_location("mv", ROOT / "analysis" / "01_fit_regression_models.py")
mv = importlib.util.module_from_spec(_s); _s.loader.exec_module(mv)

DRUGS = ["phenylephrine", "ephedrine", "vasopressin"]
FILES = {
    "xgb_sq": "oof_base_hist_xgb_{d}.npy",
    "xgb_mae": "mae_xgb_oof_base_hist_{d}.npy",
    "median": "oof_base_hist_median_{d}.npy",
    "ridge_tuned": "oof_base_hist_ridge_tuned_{d}.npy",
    "baseline_map": "oof_baseline_ridge_{d}.npy",
}


def main():
    d = mv.load()
    rows = []
    for drug in DRUGS:
        sub = d[(d.drug == drug) & (d.event_class_v12 == "iv_bolus")]
        route = sub["route"].fillna("<missing>").str.strip()
        iv = (route == "IntraVENOUS").values
        y = sub["delta_map_5min"].values.astype(float)
        for model, pat in FILES.items():
            oof = np.load(RES / pat.format(d=drug))
            a = float(np.mean([np.mean(np.abs(y - oof[r])) for r in range(oof.shape[0])]))
            b = float(np.mean([np.mean(np.abs(y[iv] - oof[r][iv])) for r in range(oof.shape[0])]))
            rows.append({"drug": drug, "model": model, "n_all": int(len(y)),
                         "n_intravenous": int(iv.sum()),
                         "n_intramuscular": int((route == "IntraMUSCULAR").sum()),
                         "n_missing_route": int((route == "<missing>").sum()),
                         "mae_all": round(a, 3), "mae_intravenous_only": round(b, 3),
                         "change": round(b - a, 4)})
    pd.DataFrame(rows).to_csv(RES / "route_audit.csv", index=False)
    print("route audit written")


if __name__ == "__main__":
    main()
