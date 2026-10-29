#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit the regression models and write out-of-fold predictions.

For each drug (phenylephrine, ephedrine, vasopressin; norepinephrine exploratory) and
each feature set (base; base + treatment history) this script fits a baseline-MAP-only
Ridge reference, a multivariable Ridge and a nested-tuned XGBoost regressor on five
independent patient-grouped 5-fold partitions. The patient->fold table depends only on
the drug, so every model and feature set within a drug is evaluated on identical events.

Preprocessing (median imputation, standardisation, rare dose-unit collapsing) is fitted
inside each training fold. History-gap features are log1p-transformed with an explicit
sentinel (-1) for "no prior dose".

Inputs:  data/event_master.parquet (host extraction output; not distributed)
Outputs (results/):
  model_comparison.csv          MAE per drug x features x model
  delta_mae.csv                 paired dMAE for the pre-specified comparisons, cluster bootstrap
  partition_summary.csv         per-partition MAE (mean/sd/range)
  oof_<feat>_<model>_<drug>.npy out-of-fold regression predictions (R x n)
  folds_<drug>.npz              the patient->fold assignment actually used
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]   # repository root
DATA = ROOT / "data"                          # event_master.parquet (not distributed)
RES = ROOT / "results"
EV = DATA / "event_master.parquet"
OUT = RES
OUT.mkdir(parents=True, exist_ok=True)

DRUGS = ["phenylephrine", "ephedrine", "vasopressin", "norepinephrine"]
EXPLORATORY = {"norepinephrine"}
N_REPEATS = 5
N_FOLDS = 5
N_BOOT = 2000
SEED = 20261027
MIN_N = 300

# Numeric base features present in the event master. `dose_unit` is handled
# separately by the in-pipeline unit encoder.
BASE_NUM = ['age', 'sex_male', 'bmi', 'asa', 'dose_rate_at_event', 'baseline_map',
            'ctx_map_mean', 'ctx_map_min', 'ctx_map_last', 'ctx_map_trend', 'ctx_map_cv',
            'ctx_time_below_65_min', 'ctx_surgery_time_min']
# History block. Gap features are supplied pre-logged with a sentinel (see load()).
HIST = ['hist_same_n_5m', 'hist_any_vaso_n_5m', 'hist_same_n_10m', 'hist_any_vaso_n_10m',
        'hist_same_n_30m', 'hist_any_vaso_n_30m', 'hist_same_first_dose',
        'hist_any_first_vaso', 'hist_min_same_log', 'hist_min_any_log',
        'hist_running_rate_same', 'hist_rate_changes_30m',
        'hist_same_bolus_dose_10m', 'hist_same_bolus_dose_30m']

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(OUT / f"fit_regression_{datetime.now():%Y%m%d_%H%M%S}.log"),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)


def load():
    d = pd.read_parquet(EV)
    d['ctx_surgery_time_min'] = d['t0_minutes_from_or_in']
    # both gap features log1p-transformed, sentinel -1 when no prior dose
    for raw, out in [('hist_min_since_same', 'hist_min_same_log'),
                     ('hist_min_since_any_vaso', 'hist_min_any_log')]:
        v = np.log1p(d[raw].clip(lower=0))
        d[out] = v.where(d[raw].notna(), -1.0)
    return d


class UnitDummies(BaseEstimator, TransformerMixin):
    """Frequency-collapse + one-hot for dose_unit, fitted on training rows only.

    Categories with >= min_count occurrences in the TRAINING fold are kept; all
    others (and unseen categories at test time) map to 'other'. Fitting inside the
    pipeline means no test-fold information enters the encoding.
    """
    def __init__(self, min_count=20):
        self.min_count = min_count

    def fit(self, X, y=None):
        s = pd.Series(np.asarray(X).ravel()).fillna('missing')
        vc = s.value_counts()
        self.keep_ = list(vc[vc >= self.min_count].index)
        self.cols_ = [f'dose_unit_{c}' for c in self.keep_] + ['dose_unit_other']
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel()).fillna('missing')
        s = s.where(s.isin(self.keep_), 'other')
        out = np.zeros((len(s), len(self.cols_)), dtype=float)
        for j, c in enumerate(self.keep_):
            out[:, j] = (s == c).values
        out[:, -1] = (s == 'other').values
        return out

    def get_feature_names_out(self, names=None):
        return np.asarray(self.cols_)


def make_folds(pats, drug):
    """Five independent partitions, SAME for every model/feature within a drug."""
    folds = []
    for r in range(N_REPEATS):
        rng = np.random.default_rng(SEED + 1000 * DRUGS.index(drug) + r)
        perm = rng.permutation(pats)
        folds.append([np.asarray(x) for x in np.array_split(perm, N_FOLDS)])
    return folds


def build_matrix(df, hist):
    num = [c for c in BASE_NUM if c in df.columns]
    if hist:
        num += [c for c in HIST if c in df.columns]
    Xnum = df[num].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return Xnum, df['dose_unit']


def xgb_space():
    return {"m__n_estimators": randint(100, 800), "m__max_depth": randint(3, 8),
            "m__learning_rate": uniform(0.01, 0.19), "m__subsample": uniform(0.6, 0.4),
            "m__colsample_bytree": uniform(0.6, 0.4), "m__reg_lambda": uniform(0.1, 4.9)}


from sklearn.compose import ColumnTransformer


def fit_fold(ftr, ytr, gtr, fte, algorithm, seed, num_cols):
    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler())]), num_cols),
         ("unit", UnitDummies(min_count=20), "dose_unit")], remainder="drop")
    if algorithm == 'ridge':
        pipe = Pipeline([("pre", pre), ("m", Ridge(alpha=1.0))])
        pipe.fit(ftr, ytr)
        return pipe.predict(fte)
    pipe = Pipeline([("pre", pre), ("m", XGBRegressor(random_state=seed, n_jobs=4))])
    inner = GroupKFold(n_splits=min(3, len(np.unique(gtr))))
    s = RandomizedSearchCV(pipe, xgb_space(), n_iter=12, scoring="neg_mean_absolute_error",
                           cv=inner.split(ftr, ytr, gtr), random_state=seed, n_jobs=1)
    s.fit(ftr, ytr, groups=gtr)
    return s.best_estimator_.predict(fte)


def run(df, drug, hist, algorithm, folds, num_cols):
    Xnum, unit = build_matrix(df, hist)
    f = Xnum.copy(); f["dose_unit"] = unit.values
    y = df['delta_map_5min'].values.astype(float); g = df['subjectid'].values
    oof_all, per_part = [], []
    for r, fold in enumerate(folds):
        oof = np.full(len(y), np.nan)
        for test_pats in fold:
            te = np.isin(g, test_pats); tr = ~te
            oof[te] = fit_fold(f.iloc[tr], y[tr], g[tr], f.iloc[te], algorithm, SEED, num_cols)
        oof_all.append(oof)
        per_part.append(float(np.mean(np.abs(y - oof))))
        log.info(f"[{drug}] hist={hist} {algorithm} repeat{r} MAE={per_part[-1]:.3f}")
    return np.stack(oof_all), per_part


def cluster_bootstrap(d_all, g):
    u = np.unique(g); idx = {p: np.where(g == p)[0] for p in u}
    rng = np.random.default_rng(SEED + 77)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.choice(u, len(u), True)
        ii = np.concatenate([idx[p] for p in s])
        boot[b] = d_all[:, ii].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    d = load()
    log.info(f"event master: {len(d)} rows; bolus analysed per drug")
    results, delta_rows, part_rows = [], [], []

    for drug in DRUGS:
        sub = d[(d.drug == drug) & (d.event_class_v12 == 'iv_bolus')].copy()
        if len(sub) < MIN_N:
            log.warning(f"[{drug}] bolus n={len(sub)} < {MIN_N}; skipped")
            continue
        role = 'exploratory' if drug in EXPLORATORY else 'primary'
        y = sub['delta_map_5min'].values.astype(float); g = sub['subjectid'].values
        pats = np.unique(g)
        folds = make_folds(pats, drug)
        np.savez(OUT / f"folds_{drug}.npz",
                 **{f"r{r}_f{k}": fold[k] for r, fold in enumerate(folds) for k in range(N_FOLDS)})

        num_base = [c for c in BASE_NUM if c in sub.columns]
        num_hist = num_base + [c for c in HIST if c in sub.columns]
        store = {}
        for feat_label, hist, ncols in [('base', False, num_base), ('base+hist', True, num_hist)]:
            for alg in ['ridge', 'xgb']:
                oof, pp = run(sub, drug, hist, alg, folds, ncols)
                store[(feat_label, alg)] = oof
                np.save(OUT / f"oof_{feat_label.replace('+','_')}_{alg}_{drug}.npy", oof)
                results.append({'drug': drug, 'role': role, 'cohort': 'bolus_iv',
                                'features': feat_label, 'model': alg,
                                'mae_mean': round(float(np.mean(pp)), 3),
                                'n': int(len(sub)), 'n_patients': int(len(pats))})
                for r, m in enumerate(pp):
                    part_rows.append({'drug': drug, 'role': role, 'features': feat_label,
                                      'model': alg, 'repeat': r, 'mae': round(m, 4),
                                      'n': int(len(sub)), 'n_patients': int(len(pats))})

        # baseline-MAP-only reference (Ridge on baseline_map), same folds
        Xb = sub[['baseline_map']].apply(pd.to_numeric, errors='coerce')
        oof_b = []
        for fold in folds:
            o = np.full(len(y), np.nan)
            for test_pats in fold:
                te = np.isin(g, test_pats); tr = ~te
                med = Xb.iloc[tr].median()
                m = Ridge(alpha=1.0).fit(Xb.iloc[tr].fillna(med), y[tr])
                o[te] = m.predict(Xb.iloc[te].fillna(med))
            oof_b.append(o)
        oof_b = np.stack(oof_b); store[('baseline', 'ridge')] = oof_b
        np.save(OUT / f"oof_baseline_ridge_{drug}.npy", oof_b)
        bpp = [float(np.mean(np.abs(y - oof_b[r]))) for r in range(N_REPEATS)]
        results.append({'drug': drug, 'role': role, 'cohort': 'bolus_iv', 'features': 'baseline',
                        'model': 'baseline_map', 'mae_mean': round(float(np.mean(bpp)), 3),
                        'n': int(len(sub)), 'n_patients': int(len(pats))})
        for r, m in enumerate(bpp):
            part_rows.append({'drug': drug, 'role': role, 'features': 'baseline',
                              'model': 'baseline_map', 'repeat': r, 'mae': round(m, 4),
                              'n': int(len(sub)), 'n_patients': int(len(pats))})

        comps = [('ridge', 'base+hist', 'ridge', 'base', 'history_effect_on_ridge'),
                 ('xgb', 'base+hist', 'xgb', 'base', 'history_effect_on_xgb'),
                 ('xgb', 'base', 'ridge', 'base', 'algorithm_effect_base'),
                 ('xgb', 'base+hist', 'ridge', 'base+hist', 'algorithm_effect_with_hist'),
                 ('xgb', 'base+hist', 'ridge', 'baseline', 'best_candidate_vs_baseline')]
        for alg_a, feat_a, alg_b, feat_b, label in comps:
            A, B = store[(feat_a, alg_a)], store[(feat_b, alg_b)]
            d_all = np.abs(y - A) - np.abs(y - B)   # (R, n); negative favours A
            lo, hi = cluster_bootstrap(d_all, g)
            per = d_all.mean(axis=1)
            delta_rows.append({'drug': drug, 'role': role, 'cohort': 'bolus_iv', 'comparison': label,
                               'A': f'{alg_a}:{feat_a}', 'B': f'{alg_b}:{feat_b}',
                               'n': int(len(sub)), 'n_patients': int(len(pats)),
                               'dmae': round(float(d_all.mean()), 3),
                               'ci_lo': round(lo, 3), 'ci_hi': round(hi, 3),
                               'sd_across_partitions': round(float(per.std(ddof=1)), 3),
                               'min_partition': round(float(per.min()), 3),
                               'max_partition': round(float(per.max()), 3)})
            log.info(f"[{drug}] {label}: {d_all.mean():+.3f} ({lo:+.3f},{hi:+.3f})")

    pd.DataFrame(results).to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(OUT / "delta_mae.csv", index=False)
    pd.DataFrame(part_rows).to_csv(OUT / "partition_summary.csv", index=False)
    log.info("done")


if __name__ == "__main__":
    main()
