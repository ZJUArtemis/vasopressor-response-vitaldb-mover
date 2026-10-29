#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit the probability models for the clinical-threshold question.

For each drug we predict P(MAP >= 65 mmHg at +4..+6 min) directly from the pre-dose
feature sets, on the same five patient partitions used by the regression models
(loaded from folds_<drug>.npz). Two learners:
  - L2-penalised logistic regression
  - XGBoost classifier, inner-fold tuned, then isotonically calibrated on an inner split

All preprocessing, tuning and probability calibration happen inside the outer training
fold. Per-fold metrics written here are superseded by the per-partition scoring in
04_score_per_partition.py; the out-of-fold probability arrays are the primary output.

Outputs (results/):
  classifier_metrics_pooled.csv
  calibration_curve_pooled.csv
  oofprob_<model>_<drug>.npy     out-of-fold predicted probabilities (R x n)
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import importlib.util
_spec = importlib.util.spec_from_file_location("fit_regression", Path(__file__).parent / "01_fit_regression_models.py")
mv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mv)

ROOT = Path(__file__).resolve().parents[1]   # repository root
DATA = ROOT / "data"                          # event_master.parquet (not distributed)
RES = ROOT / "results"
OUT = RES
DRUGS = ["phenylephrine", "ephedrine", "vasopressin", "norepinephrine"]
EXPLORATORY = {"norepinephrine"}
SEED = 20261027
N_BOOT = 2000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(OUT / f"fit_probability_{datetime.now():%Y%m%d_%H%M%S}.log"),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)


def load_folds(drug):
    z = np.load(OUT / f"folds_{drug}.npz", allow_pickle=True)
    R = max(int(k.split("_")[0][1:]) for k in z.files) + 1
    F = max(int(k.split("_")[1][1:]) for k in z.files) + 1
    return [[z[f"r{r}_f{k}"] for k in range(F)] for r in range(R)]


def preproc(num_cols):
    return ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler())]), num_cols),
         ("unit", mv.UnitDummies(min_count=20), "dose_unit")], remainder="drop")


def fit_fold(ftr, ytr, gtr, fte, model, num_cols):
    pre = preproc(num_cols)
    if model == 'logit':
        pipe = Pipeline([("pre", pre),
                         ("m", LogisticRegression(penalty="l2", C=1.0, max_iter=2000))])
        pipe.fit(ftr, ytr)
        return pipe.predict_proba(fte)[:, 1]
    base = Pipeline([("pre", pre), ("m", XGBClassifier(
        random_state=SEED, n_jobs=4, eval_metric="logloss"))])
    space = {"m__n_estimators": randint(100, 800), "m__max_depth": randint(3, 8),
             "m__learning_rate": uniform(0.01, 0.19), "m__subsample": uniform(0.6, 0.4),
             "m__colsample_bytree": uniform(0.6, 0.4), "m__reg_lambda": uniform(0.1, 4.9)}
    inner = GroupKFold(n_splits=min(3, len(np.unique(gtr))))
    s = RandomizedSearchCV(base, space, n_iter=12, scoring="neg_log_loss",
                           cv=inner.split(ftr, ytr, gtr), random_state=SEED, n_jobs=1)
    s.fit(ftr, ytr, groups=gtr)
    # isotonic calibration on an inner patient-grouped split of the training fold.
    # Pass precomputed grouped split indices (sklearn 1.7 does not route `groups`
    # to the internal CV unless metadata routing is enabled).
    cal_splits = list(GroupKFold(n_splits=min(3, len(np.unique(gtr)))).split(ftr, ytr, gtr))
    cal = CalibratedClassifierCV(s.best_estimator_, method="isotonic", cv=cal_splits)
    cal.fit(ftr, ytr)
    return cal.predict_proba(fte)[:, 1]


def calib_logodds(y, p):
    """Calibration intercept and slope from logistic recalibration on log-odds."""
    eps = 1e-6
    lo = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(penalty=None, max_iter=2000).fit(lo.reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def main():
    d = mv.load()
    rows, curve, subg = [], [], []
    rng = np.random.default_rng(SEED + 5)
    for drug in DRUGS:
        sub = d[(d.drug == drug) & (d.event_class_v12 == 'iv_bolus')].copy()
        if len(sub) < mv.MIN_N:
            continue
        role = 'exploratory' if drug in EXPLORATORY else 'primary'
        y_cont = sub['delta_map_5min'].values.astype(float)
        base_map = sub['baseline_map'].values.astype(float)
        y = ((base_map + y_cont) >= 65).astype(int)   # attain MAP>=65 at 5 min
        g = sub['subjectid'].values
        folds = load_folds(drug)
        num_base = [c for c in mv.BASE_NUM if c in sub.columns]
        num_hist = num_base + [c for c in mv.HIST if c in sub.columns]
        f_base = sub[num_base].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        f_hist = sub[num_hist].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        for fr in (f_base, f_hist):
            fr["dose_unit"] = sub['dose_unit'].values

        for feat_label, fr, ncols in [('base+hist', f_hist, num_hist)]:
            for model in ['logit', 'xgb']:
                probs = []
                for fold in folds:
                    oof = np.full(len(y), np.nan)
                    for test_pats in fold:
                        te = np.isin(g, test_pats); tr = ~te
                        oof[te] = fit_fold(fr.iloc[tr], y[tr], g[tr], fr.iloc[te], model, ncols)
                    probs.append(oof)
                probs = np.stack(probs)
                np.save(OUT / f"oofprob_{model}_{drug}.npy", probs)
                p = probs.mean(axis=0)   # stability-summary probability (labelled as such)

                auroc = roc_auc_score(y, p); auprc = average_precision_score(y, p)
                brier = brier_score_loss(y, p)
                cst_prev = brier_score_loss(y, np.full_like(p, y.mean()))  # constant-prevalence reference
                inter, slope = calib_logodds(y, p)
                yhat = (p >= 0.5).astype(int)
                tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
                fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())

                # patient-cluster bootstrap for AUROC and Brier
                u = np.unique(g); idx = {q: np.where(g == q)[0] for q in u}
                ab, bb = [], []
                for _ in range(1000):
                    s = rng.choice(u, len(u), True); ii = np.concatenate([idx[q] for q in s])
                    if len(np.unique(y[ii])) < 2:
                        continue
                    ab.append(roc_auc_score(y[ii], p[ii])); bb.append(brier_score_loss(y[ii], p[ii]))
                rows.append({'drug': drug, 'role': role, 'cohort': 'bolus_iv', 'features': feat_label,
                             'model': model, 'n': int(len(y)), 'n_patients': int(len(u)),
                             'prevalence': round(float(y.mean()), 3),
                             'auroc': round(auroc, 3), 'auroc_lo': round(float(np.percentile(ab, 2.5)), 3),
                             'auroc_hi': round(float(np.percentile(ab, 97.5)), 3),
                             'auprc': round(auprc, 3),
                             'brier': round(brier, 4), 'brier_lo': round(float(np.percentile(bb, 2.5)), 4),
                             'brier_hi': round(float(np.percentile(bb, 97.5)), 4),
                             'brier_constant_prevalence': round(float(cst_prev), 4),
                             'calib_intercept': round(inter, 3), 'calib_slope': round(slope, 3),
                             'sensitivity': round(tp / (tp + fn), 3) if tp + fn else np.nan,
                             'specificity': round(tn / (tn + fp), 3) if tn + fp else np.nan,
                             'ppv': round(tp / (tp + fp), 3) if tp + fp else np.nan,
                             'npv': round(tn / (tn + fn), 3) if tn + fn else np.nan})
                log.info(f"[{drug}] {model}: AUROC={auroc:.3f} AUPRC={auprc:.3f} Brier={brier:.4f} "
                         f"(const {cst_prev:.4f}) slope={slope:.3f}")
                # calibration curve (deciles of predicted prob) for the XGB model
                if model == 'xgb':
                    q = pd.qcut(p, 10, labels=False, duplicates='drop')
                    for b in np.unique(q):
                        mb = q == b
                        curve.append({'drug': drug, 'bin': int(b), 'n': int(mb.sum()),
                                      'pred_mean': round(float(p[mb].mean()), 4),
                                      'obs_rate': round(float(y[mb].mean()), 4)})
                    # below-65 subgroup: the patients in whom the decision matters.
                    # threshold chosen on the whole cohort to match observed attainment,
                    # then specificity (correctly flagging non-attainers) read in the subgroup.
                    thr = float(np.quantile(p, 1.0 - y.mean()))
                    sm = base_map < 65
                    if sm.sum() >= 50 and len(np.unique(y[sm])) == 2:
                        ysb, psb = y[sm], p[sm]
                        yhsb = (psb >= thr).astype(int)
                        tnb = int(((yhsb == 0) & (ysb == 0)).sum())
                        fpb = int(((yhsb == 1) & (ysb == 0)).sum())
                        tpb = int(((yhsb == 1) & (ysb == 1)).sum())
                        fnb = int(((yhsb == 0) & (ysb == 1)).sum())
                        subg.append({'drug': drug, 'subgroup': 'baseline_map_lt65',
                                     'n': int(sm.sum()), 'n_patients': int(np.unique(g[sm]).size),
                                     'observed_attainment': round(float(ysb.mean()), 3),
                                     'auroc': round(float(roc_auc_score(ysb, psb)), 3),
                                     'specificity': round(tnb / (tnb + fpb), 3) if tnb + fpb else np.nan,
                                     'sensitivity': round(tpb / (tpb + fnb), 3) if tpb + fnb else np.nan,
                                     'decision_threshold': round(thr, 3)})

    pd.DataFrame(rows).to_csv(OUT / "classifier_metrics_pooled.csv", index=False)
    pd.DataFrame(curve).to_csv(OUT / "calibration_curve_pooled.csv", index=False)
    pd.DataFrame(subg).to_csv(OUT / "classifier_subgroup_pooled.csv", index=False)
    log.info("done")


if __name__ == "__main__":
    main()
