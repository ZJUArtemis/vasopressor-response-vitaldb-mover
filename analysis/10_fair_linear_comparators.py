#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit linear comparators matched to the evaluation loss, and a baseline-MAP-only
probability model, on the same patient folds as the primary models.

Part A (regression): Ridge with the penalty chosen by inner patient-grouped cross-validation
scored on mean absolute error, and a median (tau=0.5) quantile regression, each with the base
feature set and with the treatment-history block. Writes out-of-fold predictions and the paired
dMAE comparisons against XGBoost.

Part B (probability): a logistic regression on pre-dose MAP alone, the reference a clinician
already has, and paired AUROC and Brier differences of the full classifiers against it, for the
whole cohort and the subgroup starting below 65 mmHg.

Inputs:  data/event_master.parquet, folds and out-of-fold arrays written by
         01_fit_regression_models.py and 02_fit_probability_models.py
Outputs (results/): delta_mae_fair.csv, model_comparison_linear.csv, ridge_alphas.csv,
         classifier_vs_maponly.csv, classifier_paired_diff.csv, and out-of-fold arrays.
"""
import logging
import importlib.util
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, QuantileRegressor, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
V14 = RES
OUT = RES
OUT.mkdir(parents=True, exist_ok=True)
_spec = importlib.util.spec_from_file_location("mv", ROOT / "analysis" / "01_fit_regression_models.py")
mv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mv)

DRUGS = mv.DRUGS; EXPLORATORY = mv.EXPLORATORY; SEED = mv.SEED
N_BOOT = 2000; N_BOOT_SUB = 1000
RIDGE_GRID = np.logspace(-2, 3, 11)        # 0.01 .. 1000
N_JOBS = 6

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(OUT / f"fair_comparators_{datetime.now():%Y%m%d_%H%M%S}.log"),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)


def load_folds(drug):
    z = np.load(V14 / f"folds_{drug}.npz", allow_pickle=True)
    R = max(int(k.split("_")[0][1:]) for k in z.files) + 1
    F = max(int(k.split("_")[1][1:]) for k in z.files) + 1
    return [[z[f"r{r}_f{k}"] for k in range(F)] for r in range(R)]


def preproc(num_cols):
    return ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler())]), num_cols),
         ("unit", mv.UnitDummies(min_count=20), "dose_unit")], remainder="drop")


def fit_ridge_tuned(ftr, ytr, gtr, fte, num_cols):
    inner = list(GroupKFold(n_splits=min(3, len(np.unique(gtr)))).split(ftr, ytr, gtr))
    best, best_mae = None, np.inf
    for a in RIDGE_GRID:
        maes = []
        for itr, ite in inner:
            p = Pipeline([("pre", preproc(num_cols)), ("m", Ridge(alpha=float(a)))])
            p.fit(ftr.iloc[itr], ytr[itr])
            maes.append(np.mean(np.abs(ytr[ite] - p.predict(ftr.iloc[ite]))))
        m = float(np.mean(maes))
        if m < best_mae:
            best_mae, best = m, float(a)
    p = Pipeline([("pre", preproc(num_cols)), ("m", Ridge(alpha=best))]).fit(ftr, ytr)
    return p.predict(fte), best


def fit_median(ftr, ytr, fte, num_cols):
    p = Pipeline([("pre", preproc(num_cols)),
                  ("m", QuantileRegressor(quantile=0.5, alpha=1e-3, solver="highs"))])
    p.fit(ftr, ytr)
    return p.predict(fte)


def _one_fold(model, f, y, g, te, num_cols):
    tr = ~te
    if model == "ridge_tuned":
        pred, a = fit_ridge_tuned(f.iloc[tr], y[tr], g[tr], f.iloc[te], num_cols)
        return te, pred, a
    pred = fit_median(f.iloc[tr], y[tr], f.iloc[te], num_cols)
    return te, pred, np.nan


def run_reg(sub, drug, hist, model, folds, num_cols):
    Xnum, unit = mv.build_matrix(sub, hist)
    f = Xnum.copy(); f["dose_unit"] = unit.values
    y = sub["delta_map_5min"].values.astype(float); g = sub["subjectid"].values
    jobs = [(r, np.isin(g, test_pats)) for r, fold in enumerate(folds) for test_pats in fold]
    res = Parallel(n_jobs=N_JOBS)(delayed(_one_fold)(model, f, y, g, te, num_cols) for _, te in jobs)
    oof = np.full((len(folds), len(y)), np.nan); alphas = []
    for (r, _), (te, pred, a) in zip(jobs, res):
        oof[r, te] = pred; alphas.append(a)
    pp = [float(np.mean(np.abs(y - oof[r]))) for r in range(len(folds))]
    log.info(f"[{drug}] hist={hist} {model} MAE per partition {np.round(pp,3).tolist()}"
             + (f" | alpha median {np.nanmedian(alphas):.3g} range {np.nanmin(alphas):.3g}-{np.nanmax(alphas):.3g}" if model=="ridge_tuned" else ""))
    return oof, pp, alphas


def cluster_bootstrap(d_all, g, n=N_BOOT, seed_off=77):
    u = np.unique(g); idx = {p: np.where(g == p)[0] for p in u}
    rng = np.random.default_rng(SEED + seed_off)
    boot = np.empty(n)
    for b in range(n):
        s = rng.choice(u, len(u), True)
        ii = np.concatenate([idx[p] for p in s])
        boot[b] = d_all[:, ii].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ----------------------------------------------------------------------------- Part B
def calib_logodds(y, p):
    eps = 1e-6
    lo = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(penalty=None, max_iter=2000).fit(lo.reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def fit_logit_map(sub, y, g, folds):
    Xb = sub[["baseline_map"]].apply(pd.to_numeric, errors="coerce")
    probs = []
    for fold in folds:
        o = np.full(len(y), np.nan)
        for test_pats in fold:
            te = np.isin(g, test_pats); tr = ~te
            med = Xb.iloc[tr].median()
            m = LogisticRegression(penalty=None, max_iter=2000).fit(Xb.iloc[tr].fillna(med), y[tr])
            o[te] = m.predict_proba(Xb.iloc[te].fillna(med))[:, 1]
        probs.append(o)
    return np.stack(probs)


def perrep(metric, y, P, mask=None):
    vals = []
    for r in range(P.shape[0]):
        m = np.ones(len(y), bool) if mask is None else mask
        vals.append(metric(y[m], P[r][m]))
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def perrep_ci(metric, y, P, g, mask=None, n=N_BOOT, seed_off=23):
    m = np.ones(len(y), bool) if mask is None else mask
    u = np.unique(g[m]); idx = {p: np.where((g == p) & m)[0] for p in u}
    rng = np.random.default_rng(SEED + seed_off)
    out = np.empty(n)
    for b in range(n):
        ii = np.concatenate([idx[p] for p in rng.choice(u, len(u), True)])
        out[b] = np.mean([metric(y[ii], P[r][ii]) for r in range(P.shape[0])])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_perrep_ci(metric, y, PA, PB, g, mask=None, n=N_BOOT, seed_off=31):
    """CI for mean over partitions of metric(A) - metric(B), same resampled patients."""
    m = np.ones(len(y), bool) if mask is None else mask
    u = np.unique(g[m]); idx = {p: np.where((g == p) & m)[0] for p in u}
    rng = np.random.default_rng(SEED + seed_off)
    out = np.empty(n)
    for b in range(n):
        ii = np.concatenate([idx[p] for p in rng.choice(u, len(u), True)])
        out[b] = np.mean([metric(y[ii], PA[r][ii]) - metric(y[ii], PB[r][ii]) for r in range(PA.shape[0])])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def brier(y, p): return float(brier_score_loss(y, p))
def auroc(y, p): return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
def auprc(y, p): return float(average_precision_score(y, p))


def main():
    d = mv.load()
    delta_rows, mc_rows, alpha_rows, clf_rows, clf_cmp = [], [], [], [], []
    for drug in DRUGS:
        sub = d[(d.drug == drug) & (d.event_class_v12 == "iv_bolus")].copy()
        if len(sub) < mv.MIN_N:
            continue
        role = "exploratory" if drug in EXPLORATORY else "primary"
        y = sub["delta_map_5min"].values.astype(float); g = sub["subjectid"].values
        folds = load_folds(drug)
        num_base = [c for c in mv.BASE_NUM if c in sub.columns]
        num_hist = num_base + [c for c in mv.HIST if c in sub.columns]
        log.info(f"=== {drug} n={len(sub)} patients={len(np.unique(g))} ===")

        # ---------------- Part A: regression comparators ----------------
        store = {}
        for feat_label, hist, ncols in [("base", False, num_base), ("base+hist", True, num_hist)]:
            fl = feat_label.replace("+", "_")
            store[(feat_label, "ridge")] = np.load(V14 / f"oof_{fl}_ridge_{drug}.npy")
            store[(feat_label, "xgb")] = np.load(V14 / f"oof_{fl}_xgb_{drug}.npy")
            for model in ["ridge_tuned", "median"]:
                oof, pp, alphas = run_reg(sub, drug, hist, model, folds, ncols)
                store[(feat_label, model)] = oof
                np.save(OUT / f"oof_{fl}_{model}_{drug}.npy", oof)
                mc_rows.append({"drug": drug, "role": role, "features": feat_label, "model": model,
                                "mae_mean": round(float(np.mean(pp)), 3), "mae_min": round(min(pp), 3),
                                "mae_max": round(max(pp), 3), "n": int(len(sub)), "n_patients": int(len(np.unique(g)))})
                if model == "ridge_tuned":
                    for a in alphas:
                        alpha_rows.append({"drug": drug, "features": feat_label, "alpha": a})
        store[("baseline", "ridge")] = np.load(V14 / f"oof_baseline_ridge_{drug}.npy")

        comps = [
            ("ridge_tuned", "base", "ridge", "base", "tuned_vs_fixed_ridge_base"),
            ("ridge_tuned", "base+hist", "ridge", "base+hist", "tuned_vs_fixed_ridge_hist"),
            ("ridge_tuned", "base+hist", "ridge_tuned", "base", "history_effect_on_ridge_tuned"),
            ("median", "base+hist", "median", "base", "history_effect_on_median"),
            ("xgb", "base", "ridge_tuned", "base", "algorithm_effect_base_vs_ridge_tuned"),
            ("xgb", "base+hist", "ridge_tuned", "base+hist", "algorithm_effect_hist_vs_ridge_tuned"),
            ("xgb", "base", "median", "base", "algorithm_effect_base_vs_median"),
            ("xgb", "base+hist", "median", "base+hist", "algorithm_effect_hist_vs_median"),
            ("median", "base", "ridge_tuned", "base", "median_vs_ridge_tuned_base"),
            ("ridge_tuned", "base+hist", "ridge", "baseline", "ridge_tuned_hist_vs_baseline"),
            ("median", "base+hist", "ridge", "baseline", "median_hist_vs_baseline"),
        ]
        for alg_a, feat_a, alg_b, feat_b, label in comps:
            A, B = store[(feat_a, alg_a)], store[(feat_b, alg_b)]
            d_all = np.abs(y - A) - np.abs(y - B)
            lo, hi = cluster_bootstrap(d_all, g)
            per = d_all.mean(axis=1)
            delta_rows.append({"drug": drug, "role": role, "comparison": label,
                               "A": f"{alg_a}:{feat_a}", "B": f"{alg_b}:{feat_b}",
                               "n": int(len(sub)), "n_patients": int(len(np.unique(g))),
                               "dmae": round(float(d_all.mean()), 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                               "sd_across_partitions": round(float(per.std(ddof=1)), 3),
                               "min_partition": round(float(per.min()), 3), "max_partition": round(float(per.max()), 3)})
            log.info(f"[{drug}] {label}: {d_all.mean():+.3f} ({lo:+.3f},{hi:+.3f})")

        # ---------------- Part B: baseline-MAP-only classifier ----------------
        base_map = sub["baseline_map"].values.astype(float)
        yb = ((base_map + y) >= 65).astype(int)
        smask = base_map < 65
        P = {"logit_map": fit_logit_map(sub, yb, g, folds),
             "logit": np.load(V14 / f"oofprob_logit_{drug}.npy"),
             "xgb": np.load(V14 / f"oofprob_xgb_{drug}.npy")}
        np.save(OUT / f"oofprob_logit_map_{drug}.npy", P["logit_map"])
        # OOF prevalence null Brier (train-fold prevalence), per partition
        prev_b = []
        for fold in folds:
            pv = np.full(len(yb), np.nan)
            for test_pats in fold:
                te = np.isin(g, test_pats); pv[te] = yb[~te].mean()
            prev_b.append(brier(yb, pv))
        prev_b = float(np.mean(prev_b))
        for scope, mask, nb in [("whole", None, N_BOOT), ("below65", smask, N_BOOT_SUB)]:
            for model, Pm in P.items():
                au = perrep(auroc, yb, Pm, mask); au_ci = perrep_ci(auroc, yb, Pm, g, mask, nb, 41)
                ap = perrep(auprc, yb, Pm, mask)
                br = perrep(brier, yb, Pm, mask); br_ci = perrep_ci(brier, yb, Pm, g, mask, nb, 43)
                ci = perrep(lambda a, b: calib_logodds(a, b)[0], yb, Pm, mask)
                cs = perrep(lambda a, b: calib_logodds(a, b)[1], yb, Pm, mask)
                skill_prev = 1 - br[0] / prev_b if scope == "whole" else np.nan
                br_map = perrep(brier, yb, P["logit_map"], mask)[0]
                clf_rows.append({"drug": drug, "role": role, "scope": scope, "model": model,
                                 "n": int(len(yb) if mask is None else mask.sum()),
                                 "n_patients": int(np.unique(g if mask is None else g[mask]).size),
                                 "prevalence": round(float(yb.mean() if mask is None else yb[mask].mean()), 3),
                                 "auroc_mean": round(au[0], 3), "auroc_lo": round(au_ci[0], 3), "auroc_hi": round(au_ci[1], 3),
                                 "auprc_mean": round(ap[0], 3),
                                 "brier_mean": round(br[0], 4), "brier_lo": round(br_ci[0], 4), "brier_hi": round(br_ci[1], 4),
                                 "calib_intercept_mean": round(ci[0], 3), "calib_slope_mean": round(cs[0], 3),
                                 "brier_skill_vs_prevalence": round(skill_prev, 4) if scope == "whole" else np.nan,
                                 "brier_skill_vs_maponly": round(1 - br[0] / br_map, 4)})
            for model in ["logit", "xgb"]:
                for mname, metric, so in [("auroc", auroc, 51), ("brier", brier, 53)]:
                    dm = perrep(lambda a, b: 0.0, yb, P[model], mask)  # placeholder
                    diff = np.mean([metric(yb[mask] if mask is not None else yb,
                                           (P[model][r][mask] if mask is not None else P[model][r])) -
                                    metric(yb[mask] if mask is not None else yb,
                                           (P["logit_map"][r][mask] if mask is not None else P["logit_map"][r]))
                                    for r in range(P[model].shape[0])])
                    lo, hi = paired_perrep_ci(metric, yb, P[model], P["logit_map"], g, mask, nb, so)
                    clf_cmp.append({"drug": drug, "role": role, "scope": scope, "model": model,
                                    "metric": mname, "diff_vs_maponly": round(float(diff), 4),
                                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
                    log.info(f"[{drug}] {scope} {model}-logit_map {mname}: {diff:+.4f} ({lo:+.4f},{hi:+.4f})")

    pd.DataFrame(mc_rows).to_csv(OUT / "model_comparison_linear.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(OUT / "delta_mae.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(OUT / "ridge_alphas.csv", index=False)
    pd.DataFrame(clf_rows).to_csv(OUT / "classifier_vs_maponly.csv", index=False)
    pd.DataFrame(clf_cmp).to_csv(OUT / "classifier_paired_diff.csv", index=False)
    log.info("fair linear comparators done")


if __name__ == "__main__":
    main()
