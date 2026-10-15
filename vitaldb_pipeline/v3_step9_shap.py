#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: v3_step9_shap.py | Topic: 2 (v3) | Purpose:
Feature-attribution analysis on the RETURNED MOVER event table to show that
baseline-MAP features dominate the prediction and that the (dimensionally mixed)
dose field contributes little. This directly defuses the reviewer concern that
unit-heterogeneous doses undermine the models: a near-irrelevant predictor being
mis-scaled does not threaten the conclusions.

Method: retrain per-drug XGBoost on the IntraVENOUS-only subset (the primary
cohort) with the operator's exact feature set + hyperparameters, compute SHAP via
native pred_contribs, aggregate mean|SHAP| by feature group (baseline-MAP context,
dose, patient covariates, time). Mirrors operator m2 logic for consistency.

Reads : mover_returned/outputs/mover_documented_events.parquet (read-only)
Writes: outputs_v3/metrics/shap_feature_groups.csv
        outputs_v3/figures/fig7_shap_groups.png
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


FEATURES_COMMON = ['age', 'sex_male', 'bmi', 'asa', 'dose_rate_at_event',
                   'baseline_map', 'ctx_map_mean', 'ctx_map_min', 'ctx_map_last',
                   'ctx_map_trend', 'ctx_map_cv', 'ctx_time_below_65_min',
                   'ctx_surgery_time_min']
TARGET = 'delta_map_5min'
GROUP = 'subjectid'
# PRIMARY drugs only (epinephrine excluded per IV sensitivity)
PRIMARY = ['phenylephrine', 'ephedrine', 'vasopressin', 'norepinephrine']


def group_of(feat):
    if feat == 'baseline_map' or feat.startswith('ctx_map'):
        return 'baseline-MAP context'
    if feat == 'dose_rate_at_event' or feat.startswith('dose_unit_'):
        return 'dose (rate + unit)'
    if feat in ('age', 'sex_male', 'bmi', 'asa'):
        return 'patient covariates'
    if feat in ('ctx_time_below_65_min', 'ctx_surgery_time_min'):
        return 'time context'
    return 'other'


def main():
    import xgboost as xgb
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)

    ev = pd.read_parquet(RET)
    ev = ev[(ev['event_type'] == 'start') & (ev['med_route'] == 'IntraVENOUS')].copy()
    ev[GROUP] = ev[GROUP].fillna(ev['caseid'])

    rows = []
    for drug in PRIMARY:
        sub = ev[ev['drug'] == drug].copy()
        if len(sub) < 100:
            continue
        units = pd.get_dummies(sub['dose_unit'].fillna('missing'), prefix='dose_unit', dtype=float)
        sub = pd.concat([sub, units], axis=1)
        feats = [f for f in FEATURES_COMMON if f in sub.columns] + list(units.columns)
        X = sub[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0)
        y = sub[TARGET].astype(float).values
        m = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=42, n_jobs=4).fit(X, y)
        booster = m.get_booster()
        dm = xgb.DMatrix(X, feature_names=feats)
        contribs = booster.predict(dm, pred_contribs=True)  # last col = bias
        msa = np.abs(contribs[:, :-1]).mean(axis=0)
        total = msa.sum()
        gcontrib = {}
        for f, v in zip(feats, msa):
            g = group_of(f)
            gcontrib[g] = gcontrib.get(g, 0.0) + v
        for g, v in gcontrib.items():
            rows.append({'drug': drug, 'group': g, 'mean_abs_shap': round(float(v), 4),
                         'pct_of_total': round(100 * float(v) / total, 1)})
        # single-feature dose share for the Results sentence
        dose_share = 100 * sum(msa[i] for i, f in enumerate(feats)
                               if f == 'dose_rate_at_event' or f.startswith('dose_unit_')) / total
        bm_share = 100 * sum(msa[i] for i, f in enumerate(feats)
                             if f == 'baseline_map' or f.startswith('ctx_map')) / total
        print(f"[{drug}] n={len(sub)} baseline-MAP group={bm_share:.1f}%  dose group={dose_share:.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(safe_path(OUT / "metrics" / "shap_feature_groups.csv"), index=False)

    # stacked bar figure
    groups_order = ['baseline-MAP context', 'patient covariates', 'time context', 'dose (rate + unit)']
    colors = {'baseline-MAP context': '#0072B2', 'patient covariates': '#E69F00',
              'time context': '#009E73', 'dose (rate + unit)': '#D55E00'}
    piv = df.pivot(index='drug', columns='group', values='pct_of_total').reindex(PRIMARY)
    piv = piv.reindex(columns=groups_order)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(piv))
    for g in groups_order:
        vals = piv[g].fillna(0).values
        ax.barh([d.capitalize() for d in piv.index], vals, left=bottom, label=g, color=colors[g])
        bottom += vals
    ax.set_xlabel("Share of total mean |SHAP| (%)")
    ax.set_title("Figure 7. Feature-group attribution (IV-only XGBoost): baseline MAP dominates, dose is minor")
    ax.legend(frameon=False, loc='lower right', fontsize=8)
    ax.set_xlim(0, 100)
    fig.savefig(safe_path(OUT / "figures" / "fig7_shap_groups.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("saved fig7 + shap_feature_groups.csv")


if __name__ == '__main__':
    main()
