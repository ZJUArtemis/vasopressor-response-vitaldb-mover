#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: v3_step4_interpret.py | Topic: 2 (v3) | Purpose:
SHAP global + dependence per drug, and clinical subgroup performance
(ASA, age tertiles, baseline-MAP bands, bolus vs rate_increase).
Runs per cohort. Uses XGBoost native pred_contribs (robust to SHAP/XGBoost
version drift — same workaround as v2).
"""
import os, json, logging, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = Path(os.environ.get("TOPIC2_WORK", Path(__file__).resolve().parents[1]))
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
                   'ctx_map_trend', 'ctx_map_cv', 'ctx_hr_mean', 'ctx_shock_index',
                   'ctx_time_below_65_min', 'ctx_surgery_time_min', 'redose_flag']
TARGET = 'delta_map_5min'


def mae(y, p): return float(np.mean(np.abs(y - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=str(WORK / "outputs_v3/metrics/vitaldb_documented_events.parquet"))
    ap.add_argument('--cohort', default='vitaldb')
    ap.add_argument('--outdir', default=str(WORK / "outputs_v3"))
    ap.add_argument('--min_n', type=int, default=40)
    args = ap.parse_args()
    out = Path(args.outdir)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(out / "logs" / f"{ts}_v3_step4_interpret_{args.cohort}.log"),
                  logging.StreamHandler()])
    import xgboost as xgb

    ev = pd.read_parquet(args.events)
    ev = ev[ev['event_type'] == 'start'].copy()
    ev['redose_flag'] = ev['censored_repeat'].fillna(False).astype(int)
    feats = [f for f in FEATURES_COMMON if f in ev.columns]

    subgroup_rows = []
    for drug in sorted(ev['drug'].unique()):
        sub = ev[ev['drug'] == drug].copy()
        if len(sub) < args.min_n:
            logging.warning(f"[{drug}] n={len(sub)} < {args.min_n}; skip SHAP")
            continue
        X = sub[feats].apply(pd.to_numeric, errors='coerce'); X = X.fillna(X.median())
        y = sub[TARGET].astype(float).values
        m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=42, n_jobs=4).fit(X, y)
        booster = m.get_booster()
        dm = xgb.DMatrix(X, feature_names=feats)
        contribs = booster.predict(dm, pred_contribs=True)[:, :-1]  # drop bias col
        mean_abs = np.abs(contribs).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]

        # beeswarm-ish bar + top dependence
        fig, (axb, axd) = plt.subplots(1, 2, figsize=(13, 5))
        axb.barh([feats[i] for i in order][::-1], mean_abs[order][::-1], color='#4C78A8')
        axb.set_xlabel('mean |SHAP| (mmHg)'); axb.set_title(f'{drug}: feature importance (n={len(sub)})',
                                                             fontweight='bold')
        top = feats[order[0]]
        axd.scatter(X[top], contribs[:, feats.index(top)], s=12, alpha=0.5, color='#F58518')
        axd.set_xlabel(top); axd.set_ylabel(f'SHAP for {top} (mmHg)')
        axd.set_title(f'{drug}: dependence — {top}', fontweight='bold')
        axd.axhline(0, color='k', lw=0.6, alpha=0.5)
        plt.tight_layout()
        figp = safe_path(out / "figures" / f"v3_shap_{args.cohort}_{drug}.png")
        plt.savefig(figp, dpi=300, bbox_inches='tight'); plt.close()
        logging.info(f"[{drug}] SHAP fig -> {figp} | top feats: {[feats[i] for i in order[:5]]}")

        # subgroup MAE (OOF preds if available, else resubstitution with a warning)
        oofp = out / "metrics" / f"oof_predictions_{args.cohort}_{drug}.parquet"
        if oofp.exists():
            od = pd.read_parquet(oofp)
            pred = od['pred_xgb'].values; yv = od[TARGET].values
            sg = od.copy()
        else:
            pred = m.predict(X); yv = y; sg = sub.copy()
            logging.warning(f"[{drug}] OOF preds not found; subgroup MAE uses resubstitution")
        sg = sg.assign(_pred=pred, _y=yv)
        sg = sg[~np.isnan(sg['_pred'])]
        def band(v, edges, labels):
            return pd.cut(v, bins=edges, labels=labels)
        sg['asa_grp'] = np.where(pd.to_numeric(sg['asa'], errors='coerce') >= 3, 'ASA 3-4', 'ASA 1-2')
        sg['age_grp'] = band(pd.to_numeric(sg['age'], errors='coerce'),
                             [-1, 50, 65, 200], ['<50', '50-65', '>65'])
        sg['bmap_grp'] = band(sg['baseline_map'], [0, 60, 70, 300], ['<60', '60-70', '>70'])
        for col in ['asa_grp', 'age_grp', 'bmap_grp']:
            for g, gd in sg.groupby(col):
                if len(gd) >= 10:
                    subgroup_rows.append({'cohort': args.cohort, 'drug': drug, 'dimension': col,
                                          'group': str(g), 'n': len(gd),
                                          'MAE': round(mae(gd['_y'].values, gd['_pred'].values), 2)})

    sdf = pd.DataFrame(subgroup_rows)
    sp = safe_path(out / "metrics" / f"v3_subgroup_{args.cohort}.csv")
    sdf.to_csv(sp, index=False)
    logging.info(f"Saved subgroups -> {sp}\n{sdf.to_string()}")
    logging.info("=== Step 4 complete ===")


if __name__ == '__main__':
    main()
