#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sensitivity analysis for the baseline window.

Baseline MAP was the median of the 2 minutes before the dose; when that window held no
sample, the published extraction used the 10-minute pre-window median instead of excluding
the administration. This script removes the administrations that used that fallback and
recomputes mean absolute error from the saved out-of-fold predictions, which is the strict
definition stated in the Methods.

Inputs:  data/event_master.parquet; out-of-fold arrays from 01, 10, 12
Outputs (results/): strict_window_check.csv
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


def main():
    d = mv.load()
    b = d[(d.event_class_v12 == "iv_bolus") & (d.drug.isin(DRUGS))]
    rows = []
    for drug in DRUGS:
        sub = b[b.drug == drug]
        y = sub["delta_map_5min"].values.astype(float)
        fallback = (sub["baseline_map"] - sub["ctx_map_mean"]).abs() < 1e-9
        keep = ~fallback.values
        r = {"drug": drug, "n_all": int(len(y)), "n_fallback": int(fallback.sum()),
             "n_strict": int(keep.sum())}
        for name in ["xgb_sq", "xgb_mae", "median"]:
            path = RES / f"{'mae_xgb' if name == 'xgb_mae' else ''}{''}"
            files = {"xgb_sq": RES / f"oof_base_hist_xgb_{drug}.npy",
                     "xgb_mae": RES / f"mae_xgb_oof_base_hist_{drug}.npy",
                     "median": RES / f"oof_base_hist_median_{drug}.npy"}
            oof = np.load(files[name])
            r[f"mae_all_{name}"] = round(float(np.mean([np.mean(np.abs(y - oof[i])) for i in range(oof.shape[0])])), 3)
            r[f"mae_strict_{name}"] = round(float(np.mean([np.mean(np.abs(y[keep] - oof[i][keep])) for i in range(oof.shape[0])])), 3)
        rows.append(r)
    pd.DataFrame(rows).to_csv(RES / "strict_window_check.csv", index=False)
    print("strict-window check written")


if __name__ == "__main__":
    main()
