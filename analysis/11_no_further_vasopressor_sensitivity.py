#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sensitivity analysis excluding administrations followed by any further vasopressor within
the outcome window (first further record at or before +6 minutes), and the excluded complement.

Every model is refitted within each cohort on patient-grouped folds drawn by the same rule:
baseline-MAP ridge, tuned Ridge and XGBoost with each feature set, and the probability models
including the MAP-only reference. The exclusion is a selection on post-dose care, so this is a
sensitivity analysis, not an unbiased estimate of drug physiology.

Inputs:  data/event_master.parquet
Outputs (results/): sens_cohort.csv, sens_model_comparison.csv, sens_delta_mae.csv,
         sens_quality.csv, sens_classifier.csv, and sensitivity-cohort out-of-fold arrays.
"""
import logging
import importlib.util
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from xgboost import XGBClassifier
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
V14 = RES; V19 = RES
_s = importlib.util.spec_from_file_location("mv", ROOT / "analysis" / "01_fit_regression_models.py"); mv = importlib.util.module_from_spec(_s); _s.loader.exec_module(mv)
_s2 = importlib.util.spec_from_file_location("m19", V19 / "10_fair_linear_comparators.py"); m19 = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(m19)

OUT = V19
DRUGS = ["phenylephrine", "ephedrine", "vasopressin"]   # primary drugs only
SEED = mv.SEED; N_BOOT = 2000; N_BOOT_SUB = 1000; N_JOBS = 3
WINDOW_END_MIN = 6

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(OUT / f"sensitivity_{datetime.now():%Y%m%d_%H%M%S}.log"),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)


def further_mask(sub):
    a = sub["fu_other_vaso_10m_first_min"].astype(float)
    b = sub["fu_same_redose_10m_first_min"].astype(float)
    return ((a <= WINDOW_END_MIN) | (b <= WINDOW_END_MIN)).fillna(False).values


def make_folds(pats, drug, tag):
    folds = []
    for r in range(mv.N_REPEATS):
        rng = np.random.default_rng(SEED + 1000 * mv.DRUGS.index(drug) + r + (0 if tag == "sens" else 500))
        perm = rng.permutation(pats)
        folds.append([np.asarray(x) for x in np.array_split(perm, mv.N_FOLDS)])
    return folds


def _fold_reg(model, f, y, g, te, num_cols):
    tr = ~te
    if model == "ridge_tuned":
        pred, _ = m19.fit_ridge_tuned(f.iloc[tr], y[tr], g[tr], f.iloc[te], num_cols)
    else:
        pred = mv.fit_fold(f.iloc[tr], y[tr], g[tr], f.iloc[te], "xgb", SEED, num_cols)
    return te, pred


def run_reg(sub, hist, model, folds, num_cols):
    Xnum, unit = mv.build_matrix(sub, hist)
    f = Xnum.copy(); f["dose_unit"] = unit.values
    y = sub["delta_map_5min"].values.astype(float); g = sub["subjectid"].values
    jobs = [(r, np.isin(g, tp)) for r, fold in enumerate(folds) for tp in fold]
    res = Parallel(n_jobs=N_JOBS)(delayed(_fold_reg)(model, f, y, g, te, num_cols) for _, te in jobs)
    oof = np.full((len(folds), len(y)), np.nan)
    for (r, _), (te, pred) in zip(jobs, res):
        oof[r, te] = pred
    return oof


def run_baseline(sub, folds):
    y = sub["delta_map_5min"].values.astype(float); g = sub["subjectid"].values
    Xb = sub[["baseline_map"]].apply(pd.to_numeric, errors="coerce")
    oof = np.full((len(folds), len(y)), np.nan)
    for r, fold in enumerate(folds):
        for tp in fold:
            te = np.isin(g, tp); tr = ~te
            med = Xb.iloc[tr].median()
            oof[r, te] = Ridge(alpha=1.0).fit(Xb.iloc[tr].fillna(med), y[tr]).predict(Xb.iloc[te].fillna(med))
    return oof


def _fold_clf(model, f, yb, g, te, num_cols):
    tr = ~te
    pre = m19.preproc(num_cols)
    if model == "logit":
        p = Pipeline([("pre", pre), ("m", LogisticRegression(penalty="l2", C=1.0, max_iter=2000))]).fit(f.iloc[tr], yb[tr])
        return te, p.predict_proba(f.iloc[te])[:, 1]
    base = Pipeline([("pre", pre), ("m", XGBClassifier(random_state=SEED, n_jobs=2, eval_metric="logloss"))])
    space = {"m__n_estimators": randint(100, 800), "m__max_depth": randint(3, 8),
             "m__learning_rate": uniform(0.01, 0.19), "m__subsample": uniform(0.6, 0.4),
             "m__colsample_bytree": uniform(0.6, 0.4), "m__reg_lambda": uniform(0.1, 4.9)}
    gtr = g[tr]
    inner = GroupKFold(n_splits=min(3, len(np.unique(gtr))))
    s = RandomizedSearchCV(base, space, n_iter=12, scoring="neg_log_loss",
                           cv=inner.split(f.iloc[tr], yb[tr], gtr), random_state=SEED, n_jobs=1)
    s.fit(f.iloc[tr], yb[tr], groups=gtr)
    cal_splits = list(GroupKFold(n_splits=min(3, len(np.unique(gtr)))).split(f.iloc[tr], yb[tr], gtr))
    cal = CalibratedClassifierCV(s.best_estimator_, method="isotonic", cv=cal_splits).fit(f.iloc[tr], yb[tr])
    return te, cal.predict_proba(f.iloc[te])[:, 1]


def run_clf(sub, yb, model, folds, num_cols):
    Xnum, unit = mv.build_matrix(sub, True)
    f = Xnum.copy(); f["dose_unit"] = unit.values
    g = sub["subjectid"].values
    jobs = [(r, np.isin(g, tp)) for r, fold in enumerate(folds) for tp in fold]
    res = Parallel(n_jobs=N_JOBS if model == "logit" else 3)(delayed(_fold_clf)(model, f, yb, g, te, num_cols) for _, te in jobs)
    P = np.full((len(folds), len(yb)), np.nan)
    for (r, _), (te, pred) in zip(jobs, res):
        P[r, te] = pred
    return P


def r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
def mae(y, p): return float(np.mean(np.abs(y - p)))


def analyse(tag, sub, drug, rows):
    cohort_rows, mc, dm, q, clf = rows
    y = sub["delta_map_5min"].values.astype(float); g = sub["subjectid"].values
    pats = np.unique(g); folds = make_folds(pats, drug, tag)
    num_base = [c for c in mv.BASE_NUM if c in sub.columns]
    num_hist = num_base + [c for c in mv.HIST if c in sub.columns]
    store = {("baseline", "ridge"): run_baseline(sub, folds)}
    for feat, hist, nc in [("base", False, num_base), ("base+hist", True, num_hist)]:
        for model in ["ridge_tuned", "xgb"]:
            store[(feat, model)] = run_reg(sub, hist, model, folds, nc)
            np.save(OUT / f"{tag}_oof_{feat.replace('+','_')}_{model}_{drug}.npy", store[(feat, model)])
    for (feat, model), oof in store.items():
        pp = [mae(y, oof[r]) for r in range(oof.shape[0])]
        mc.append({"cohort": tag, "drug": drug, "features": feat, "model": model, "n": len(y),
                   "n_patients": len(pats), "mae_mean": round(np.mean(pp), 3), "mae_min": round(min(pp), 3), "mae_max": round(max(pp), 3)})
    comps = [("ridge_tuned", "base+hist", "ridge_tuned", "base", "history_effect_on_ridge_tuned"),
             ("xgb", "base+hist", "xgb", "base", "history_effect_on_xgb"),
             ("xgb", "base", "ridge_tuned", "base", "algorithm_effect_base_vs_ridge_tuned"),
             ("xgb", "base+hist", "ridge_tuned", "base+hist", "algorithm_effect_hist_vs_ridge_tuned"),
             ("xgb", "base+hist", "ridge", "baseline", "best_candidate_vs_baseline")]
    for aa, fa, ab, fb, label in comps:
        d_all = np.abs(y - store[(fa, aa)]) - np.abs(y - store[(fb, ab)])
        lo, hi = m19.cluster_bootstrap(d_all, g)
        per = d_all.mean(axis=1)
        dm.append({"cohort": tag, "drug": drug, "comparison": label, "A": f"{aa}:{fa}", "B": f"{ab}:{fb}",
                   "n": len(y), "n_patients": len(pats), "dmae": round(float(d_all.mean()), 3),
                   "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "sd_across_partitions": round(float(per.std(ddof=1)), 3)})
        log.info(f"[{tag}/{drug}] {label}: {d_all.mean():+.3f} ({lo:+.3f},{hi:+.3f})")
    oof = store[("base+hist", "xgb")]
    r2m = m19.perrep(r2, y, oof); r2ci = m19.perrep_ci(r2, y, oof, g, None, N_BOOT, 61)
    r2b = m19.perrep(r2, y, store[("baseline", "ridge")])
    q.append({"cohort": tag, "drug": drug, "n": len(y), "n_patients": len(pats),
              "outcome_sd": round(float(y.std(ddof=1)), 2), "observed_delta_mean": round(float(y.mean()), 2),
              "mae_mean": round(m19.perrep(mae, y, oof)[0], 3), "r2_mean": round(r2m[0], 3),
              "r2_lo": round(r2ci[0], 3), "r2_hi": round(r2ci[1], 3), "r2_baseline_only": round(r2b[0], 3)})
    # classifiers
    base_map = sub["baseline_map"].values.astype(float)
    yb = ((base_map + y) >= 65).astype(int); smask = base_map < 65
    P = {"logit_map": m19.fit_logit_map(sub, yb, g, folds),
         "logit": run_clf(sub, yb, "logit", folds, num_hist),
         "xgb": run_clf(sub, yb, "xgb", folds, num_hist)}
    for model, Pm in P.items():
        np.save(OUT / f"{tag}_oofprob_{model}_{drug}.npy", Pm)
    for scope, mask, nb in [("whole", None, N_BOOT), ("below65", smask, N_BOOT_SUB)]:
        for model, Pm in P.items():
            au = m19.perrep(m19.auroc, yb, Pm, mask); auci = m19.perrep_ci(m19.auroc, yb, Pm, g, mask, nb, 71)
            br = m19.perrep(m19.brier, yb, Pm, mask); brci = m19.perrep_ci(m19.brier, yb, Pm, g, mask, nb, 73)
            cs = m19.perrep(lambda a, b: m19.calib_logodds(a, b)[1], yb, Pm, mask)
            brmap = m19.perrep(m19.brier, yb, P["logit_map"], mask)[0]
            clf.append({"cohort": tag, "drug": drug, "scope": scope, "model": model,
                        "n": int(len(yb) if mask is None else mask.sum()),
                        "prevalence": round(float(yb.mean() if mask is None else yb[mask].mean()), 3),
                        "auroc_mean": round(au[0], 3), "auroc_lo": round(auci[0], 3), "auroc_hi": round(auci[1], 3),
                        "brier_mean": round(br[0], 4), "brier_lo": round(brci[0], 4), "brier_hi": round(brci[1], 4),
                        "calib_slope_mean": round(cs[0], 3), "brier_skill_vs_maponly": round(1 - br[0] / brmap, 4)})
    cohort_rows.append({"cohort": tag, "drug": drug, "n": len(y), "n_patients": len(pats)})


def main():
    d = mv.load()
    rows = ([], [], [], [], [])
    for drug in DRUGS:
        sub = d[(d.drug == drug) & (d.event_class_v12 == "iv_bolus")].copy()
        excl = further_mask(sub)
        log.info(f"=== {drug}: n={len(sub)}, further<=+{WINDOW_END_MIN}min excluded={excl.sum()} ({excl.mean()*100:.1f}%), sens n={(~excl).sum()}")
        analyse("sens", sub[~excl].copy(), drug, rows)
        analyse("excluded", sub[excl].copy(), drug, rows)
    pd.DataFrame(rows[0]).to_csv(OUT / "sens_cohort.csv", index=False)
    pd.DataFrame(rows[1]).to_csv(OUT / "sens_model_comparison.csv", index=False)
    pd.DataFrame(rows[2]).to_csv(OUT / "sens_delta_mae.csv", index=False)
    pd.DataFrame(rows[3]).to_csv(OUT / "sens_quality.csv", index=False)
    pd.DataFrame(rows[4]).to_csv(OUT / "sens_classifier.csv", index=False)
    log.info("no-further-vasopressor sensitivity done")


if __name__ == "__main__":
    main()
