#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Score every reported performance measure under one convention, from the saved
out-of-fold arrays (no retraining).

Per-partition scoring: each of the five patient partitions is one complete out-of-fold
vector in which every event is predicted once. Every metric is computed on each
partition separately and reported as the across-partition mean with its range and a
patient-cluster bootstrap CI whose resampling target is that same across-partition mean.
This applies to regression MAE/RMSE/R2/calibration slope/large-error proportions,
classifier AUROC/AUPRC/Brier/calibration intercept and slope, the below-65 subgroup,
the pre-dose MAP strata, the descriptive operating points and the follow-up strata.

The only exception is the binned reliability / regression-decile curve, which needs one
prediction per event to draw; it uses the five-partition mean prediction and is labelled
a stability summary.

The difference-in-differences (history effect under XGBoost minus history effect under
Ridge) is computed per partition and per event from the four regression arrays, with a
single paired bootstrap, so that it equals the difference of the two history effects.

Bootstrap resamples: N_BOOT for whole-cohort quantities; N_BOOT_SUB for calibration
parameters, subgroup and strata; N_BOOT_BIN for per-bin reliability points.

Outputs (results/):
  regression_perpartition.csv, regression_quality.csv, map_strata.csv,
  regression_calibration.csv, baseline_r2.csv, did.csv,
  classifier_perpartition.csv, classifier_metrics.csv, classifier_subgroup.csv,
  calibration_curve.csv, operating_points.csv, prob_baseline.csv, followup_strata.csv
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]   # repository root
DATA = ROOT / "data"                          # event_master.parquet (not distributed)
RES = ROOT / "results"
EV = DATA / "event_master.parquet"
V14 = RES
OUT = RES
OUT.mkdir(parents=True, exist_ok=True)

DRUGS = ["phenylephrine", "ephedrine", "vasopressin", "norepinephrine"]
EXPLORATORY = {"norepinephrine"}
SEED = 20261027
N_BOOT = 2000
N_BOOT_SUB = 1000      # strata / subgroup / calibration-parameter CIs
N_BOOT_BIN = 500       # per-bin reliability points
OP_GRID = [0.50, 0.60, 0.70, 0.80, 0.90]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("score_per_partition")


# ----------------------------------------------------------------------------- loaders
def load_events():
    return pd.read_parquet(EV)


def sub_for(d, drug):
    """Analysed bolus cohort for a drug, in the same row order used when the models were fitted, so the
    OOF columns line up."""
    return d[(d.drug == drug) & (d.event_class_v12 == "iv_bolus")].copy()


def oof_reg(drug, feat, model):
    return np.load(V14 / f"oof_{feat}_{model}_{drug}.npy")   # (R, n)


def oof_prob(drug, model):
    return np.load(V14 / f"oofprob_{model}_{drug}.npy")      # (R, n)


def load_folds(drug):
    z = np.load(V14 / f"folds_{drug}.npz", allow_pickle=True)
    R = max(int(k.split("_")[0][1:]) for k in z.files) + 1
    F = max(int(k.split("_")[1][1:]) for k in z.files) + 1
    return [[z[f"r{r}_f{k}"] for k in range(F)] for r in range(R)]


# ------------------------------------------------------------------------- metric fns
def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def abs_err_gt(thr):
    return lambda y, p: 100.0 * float(np.mean(np.abs(y - p) > thr))


def reg_calib_slope(y, p):
    if np.std(p) < 1e-9:
        return None
    return float(np.polyfit(p, y, 1)[0])


def reg_calib_intercept(y, p):
    if np.std(p) < 1e-9:
        return None
    return float(np.polyfit(p, y, 1)[1])


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def auprc(y, p):
    """Average precision via the standard sklearn implementation (handles ties properly)."""
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def roc_auc(y, p):
    """AUROC via the Mann-Whitney rank identity, with averaged ranks for ties."""
    if len(np.unique(y)) < 2:
        return None
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    avg, start = {}, 0
    for i, c in enumerate(counts):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calib_logodds(y, p):
    """Calibration intercept and slope from a single logistic recalibration on the
    log-odds of the predicted probability (unpenalised logistic of outcome on the logit,
    intercept and slope both free). Solved by Newton/IRLS in numpy; this is the same
    estimand sklearn's penalty=None fit returns but is fast enough to resample thousands
    of times in the cluster bootstrap. Falls back to sklearn if IRLS fails to converge."""
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    x = np.log(pc / (1 - pc))
    X = np.column_stack([np.ones_like(x), x])          # [intercept, slope]
    b = np.zeros(2)
    for _ in range(50):
        eta = X @ b
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        grad = X.T @ (y - mu)
        H = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            return float(b[0]), float(b[1])
    # non-convergence fallback (rare; small or separable subgroup resamples)
    lr = LogisticRegression(penalty=None, max_iter=2000).fit(x.reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def mean_range(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


# ------------------------------------------------------ per-repeat cluster-bootstrap CI
def perrep_point(metric_fn, y, oof, mask=None):
    """Across-partition mean of a per-repeat metric. Each repeat scored on its own OOF."""
    R = oof.shape[0]
    vals = []
    for r in range(R):
        yr = y if mask is None else y[mask]
        pr = oof[r] if mask is None else oof[r][mask]
        v = metric_fn(yr, pr)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def perrep_ci(metric_fn, y, oof, g, mask=None, n=N_BOOT, seed_off=23):
    """Cluster-bootstrap CI whose target is the across-partition mean of the per-repeat
    metric. Each resample draws PATIENTS with replacement once, then every repeat is
    scored on that same resampled set and averaged -- keeping the interval's estimand
    identical to the point estimate."""
    if mask is not None:
        y = y[mask]
        oof = oof[:, mask]
        g = g[mask]
    u = np.unique(g)
    idx = {q: np.where(g == q)[0] for q in u}
    R = oof.shape[0]
    rng = np.random.default_rng(SEED + seed_off)
    vals = []
    for _ in range(n):
        s = rng.choice(u, len(u), True)
        ii = np.concatenate([idx[q] for q in s])
        yi = y[ii]
        per = []
        for r in range(R):
            v = metric_fn(yi, oof[r][ii])
            if v is not None and np.isfinite(v):
                per.append(v)
        if per:
            vals.append(np.mean(per))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------------- regression
REG_METRICS = {
    "mae": mae, "rmse": rmse, "r2": r2,
    "calib_slope": reg_calib_slope, "calib_intercept": reg_calib_intercept,
    "pct_err_gt_5": abs_err_gt(5), "pct_err_gt_10": abs_err_gt(10),
    "pct_err_gt_15": abs_err_gt(15),
}


def regression_tables(d):
    """Per-repeat regression metrics for the primary candidate (XGBoost + history), the
    three-tier MAP-strata MAE (now per-repeat), and the pooled regression calibration
    curve (labelled as a stability summary)."""
    per_rows, qual_rows, strata_rows, cal_rows = [], [], [], []
    for drug in DRUGS:
        sub = sub_for(d, drug)
        if len(sub) < 300:
            continue
        role = "exploratory" if drug in EXPLORATORY else "primary"
        y = sub["delta_map_5min"].values.astype(float)
        g = sub["subjectid"].values
        base_map = sub["baseline_map"].values.astype(float)
        sd = float(np.std(y, ddof=1))
        oof = oof_reg(drug, "base_hist", "xgb")                # primary candidate
        R = oof.shape[0]

        summ = {}
        for name, fn in REG_METRICS.items():
            vals = [fn(y, oof[r]) for r in range(R)]
            m, lo, hi = mean_range(vals)
            summ[name] = (m, lo, hi)
            for r, v in enumerate(vals):
                if v is None or not np.isfinite(v):
                    continue
                per_rows.append({"drug": drug, "role": role, "model": "xgb_hist",
                                 "metric": name, "repeat": r, "value": round(float(v), 5)})

        boot = {}
        for name, fn, off in [("r2", r2, 31), ("calib_slope", reg_calib_slope, 33),
                              ("pct_err_gt_5", abs_err_gt(5), 35),
                              ("pct_err_gt_10", abs_err_gt(10), 37),
                              ("pct_err_gt_15", abs_err_gt(15), 39)]:
            boot[name] = perrep_ci(fn, y, oof, g, seed_off=off)

        qual_rows.append({
            "drug": drug, "role": role, "n": int(len(sub)),
            "n_patients": int(np.unique(g).size), "outcome_sd": round(sd, 3),
            "mae_mean": round(summ["mae"][0], 3), "mae_min": round(summ["mae"][1], 3),
            "mae_max": round(summ["mae"][2], 3), "rmse_mean": round(summ["rmse"][0], 3),
            "mae_over_sd_mean": round(summ["mae"][0] / sd, 4),
            "r2_mean": round(summ["r2"][0], 4), "r2_min": round(summ["r2"][1], 4),
            "r2_max": round(summ["r2"][2], 4),
            "r2_boot_lo": round(boot["r2"][0], 4), "r2_boot_hi": round(boot["r2"][1], 4),
            "calib_slope_mean": round(summ["calib_slope"][0], 3),
            "calib_slope_boot_lo": round(boot["calib_slope"][0], 3),
            "calib_slope_boot_hi": round(boot["calib_slope"][1], 3),
            "calib_intercept_mean": round(summ["calib_intercept"][0], 3),
            "pct_err_gt_5_mean": round(summ["pct_err_gt_5"][0], 1),
            "pct_err_gt_5_lo": round(boot["pct_err_gt_5"][0], 1),
            "pct_err_gt_5_hi": round(boot["pct_err_gt_5"][1], 1),
            "pct_err_gt_10_mean": round(summ["pct_err_gt_10"][0], 1),
            "pct_err_gt_10_lo": round(boot["pct_err_gt_10"][0], 1),
            "pct_err_gt_10_hi": round(boot["pct_err_gt_10"][1], 1),
            "pct_err_gt_15_mean": round(summ["pct_err_gt_15"][0], 1),
            "pct_err_gt_15_lo": round(boot["pct_err_gt_15"][0], 1),
            "pct_err_gt_15_hi": round(boot["pct_err_gt_15"][1], 1),
        })

        # three-tier pre-dose MAP strata (PER-REPEAT MAE within each stratum)
        for label, msk in [("lt65", base_map < 65),
                           ("65to75", (base_map >= 65) & (base_map <= 75)),
                           ("gt75", base_map > 75)]:
            if msk.sum() < 50:
                continue
            m = perrep_point(mae, y, oof, mask=msk)
            lo, hi = perrep_ci(mae, y, oof, g, mask=msk, n=N_BOOT_SUB, seed_off=41)
            strata_rows.append({"drug": drug, "role": role, "stratum": label,
                                "n": int(msk.sum()), "n_patients": int(np.unique(g[msk]).size),
                                "mae": round(m, 3), "lo": round(lo, 3), "hi": round(hi, 3),
                                "observed_delta_mean": round(float(y[msk].mean()), 2),
                                "baseline_map_mean": round(float(base_map[msk].mean()), 1)})

        # regression calibration curve: deciles of the five-repeat MEAN prediction
        # (stability summary; the only statistic that keeps the pooled prediction)
        pooled = np.nanmean(oof, axis=0)
        qd = pd.qcut(pooled, 10, labels=False, duplicates="drop")
        for b in np.unique(qd):
            mb = qd == b
            cal_rows.append({"drug": drug, "bin": int(b), "n": int(mb.sum()),
                             "pred_mean": round(float(pooled[mb].mean()), 3),
                             "obs_mean": round(float(y[mb].mean()), 3),
                             "obs_sd": round(float(y[mb].std(ddof=1)), 3)})

    pd.DataFrame(per_rows).to_csv(OUT / "regression_perpartition.csv", index=False)
    pd.DataFrame(qual_rows).to_csv(OUT / "regression_quality.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(OUT / "map_strata.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUT / "regression_calibration.csv", index=False)
    log.info("regression tables written")


def baseline_r2_table(d):
    """Per-repeat baseline-only R2 (Ridge on baseline MAP only) and its increment to the
    primary candidate's R2, so the 'adds ~0.05-0.09 R2' sentence has an explicit source."""
    rows = []
    for drug in DRUGS:
        sub = sub_for(d, drug)
        if len(sub) < 300:
            continue
        y = sub["delta_map_5min"].values.astype(float)
        base = oof_reg(drug, "baseline", "ridge")
        full = oof_reg(drug, "base_hist", "xgb")
        R = base.shape[0]
        r2_base = [r2(y, base[r]) for r in range(R)]
        r2_full = [r2(y, full[r]) for r in range(R)]
        mb, lob, hib = mean_range(r2_base)
        mf, _, _ = mean_range(r2_full)
        rows.append({"drug": drug, "n": int(len(sub)),
                     "r2_baseline_only_mean": round(mb, 4), "r2_baseline_only_min": round(lob, 4),
                     "r2_baseline_only_max": round(hib, 4), "r2_full_mean": round(mf, 4),
                     "r2_increment": round(mf - mb, 4)})
    pd.DataFrame(rows).to_csv(OUT / "baseline_r2.csv", index=False)
    log.info("baseline r2 table written")


# -------------------------------------------------------------------------------- DiD
def did_table(d):
    """Review 4.2: difference-in-differences computed PER REPEAT, PER EVENT, so it equals
    the difference of the two history effects under the identical per-repeat convention.

        D[r,i] = (|y_i - XGB_H[r,i]|  - |y_i - XGB_base[r,i]|)
               - (|y_i - Ridge_H[r,i]| - |y_i - Ridge_base[r,i]|)
        DiD    = mean over r and i of D[r,i]

    CI from a paired patient-cluster bootstrap in which the same resampled patients enter
    all four OOF arrays and all repeats together."""
    rows = []
    for drug in DRUGS:
        sub = sub_for(d, drug)
        if len(sub) < 300:
            continue
        role = "exploratory" if drug in EXPLORATORY else "primary"
        y = sub["delta_map_5min"].values.astype(float)
        g = sub["subjectid"].values
        xg_h = oof_reg(drug, "base_hist", "xgb")
        xg_b = oof_reg(drug, "base", "xgb")
        rd_h = oof_reg(drug, "base_hist", "ridge")
        rd_b = oof_reg(drug, "base", "ridge")
        R = xg_h.shape[0]
        # per-repeat, per-event DiD matrix (R, n)
        D = (np.abs(y - xg_h) - np.abs(y - xg_b)) - (np.abs(y - rd_h) - np.abs(y - rd_b))
        point = float(D.mean())                                  # mean over r and i

        u = np.unique(g)
        idx = {q: np.where(g == q)[0] for q in u}
        rng = np.random.default_rng(SEED + 51)
        boots = []
        for _ in range(N_BOOT):
            s = rng.choice(u, len(u), True)
            ii = np.concatenate([idx[q] for q in s])
            boots.append(float(D[:, ii].mean()))                 # all repeats together
        rows.append({"drug": drug, "role": role, "n": int(len(sub)),
                     "n_patients": int(len(u)), "did": round(point, 4),
                     "ci_lo": round(float(np.percentile(boots, 2.5)), 4),
                     "ci_hi": round(float(np.percentile(boots, 97.5)), 4),
                     # the two same-convention history effects, for the audit equality check
                     "hist_xgb": round(float((np.abs(y - xg_h) - np.abs(y - xg_b)).mean()), 4),
                     "hist_ridge": round(float((np.abs(y - rd_h) - np.abs(y - rd_b)).mean()), 4)})
    pd.DataFrame(rows).to_csv(OUT / "did.csv", index=False)
    log.info("DiD table written")


# -------------------------------------------------------------------------- classifier
def _calib_curve(rows, drug, scope, y, p, g):
    """Binned reliability with per-bin n, observed rate and a cluster-bootstrap CI.
    Uses the five-repeat mean probability `p` (stability summary)."""
    try:
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
    except ValueError:
        q = pd.qcut(p, 5, labels=False, duplicates="drop")
    u = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in u}
    for b in np.unique(q):
        mb = q == b
        yb = y[mb]
        # cluster bootstrap of the observed rate within this bin
        sub_g = g[mb]
        su = np.unique(sub_g)
        sidx = {k: np.where(sub_g == k)[0] for k in su}
        rng = np.random.default_rng(SEED + 73 + int(b))
        vals = []
        for _ in range(N_BOOT_BIN):
            s = rng.choice(su, len(su), True)
            ii = np.concatenate([sidx[k] for k in s])
            vals.append(float(yb[ii].mean()))
        rows.append({"drug": drug, "scope": scope, "bin": int(b), "n": int(mb.sum()),
                     "pred_mean": round(float(p[mb].mean()), 4),
                     "obs_rate": round(float(yb.mean()), 4),
                     "obs_lo": round(float(np.percentile(vals, 2.5)), 4),
                     "obs_hi": round(float(np.percentile(vals, 97.5)), 4)})


def classifier_tables(d):
    """Per-repeat classifier metrics (AUROC, AUPRC, Brier, calibration intercept/slope),
    per-repeat below-65 subgroup metrics, per-repeat descriptive operating points, a proper
    OOF training-fold prevalence baseline, and the pooled reliability curve."""
    per_rows, met_rows, sub_rows, curve_rows, op_rows, base_rows = [], [], [], [], [], []
    for drug in DRUGS:
        sub = sub_for(d, drug)
        if len(sub) < 300:
            continue
        role = "exploratory" if drug in EXPLORATORY else "primary"
        base_map = sub["baseline_map"].values.astype(float)
        y_cont = sub["delta_map_5min"].values.astype(float)
        y = ((base_map + y_cont) >= 65).astype(int)        # attain MAP>=65 at 5 min
        g = sub["subjectid"].values
        folds = load_folds(drug)
        smask = base_map < 65

        for model in ["logit", "xgb"]:
            prob = oof_prob(drug, model)                    # (R, n)
            R = prob.shape[0]
            # ---- per-repeat metrics (AUROC, AUPRC, Brier, calib intercept/slope) ----
            metr = {"auroc": roc_auc, "auprc": auprc, "brier": brier,
                    "calib_intercept": lambda a, b: calib_logodds(a, b)[0],
                    "calib_slope": lambda a, b: calib_logodds(a, b)[1]}
            summ = {}
            for name, fn in metr.items():
                vals = [fn(y, prob[r]) for r in range(R)]
                summ[name] = mean_range(vals)
                for r, v in enumerate(vals):
                    if v is None or not np.isfinite(v):
                        continue
                    per_rows.append({"drug": drug, "role": role, "model": model,
                                     "metric": name, "repeat": r, "value": round(float(v), 5)})
            au_cl = perrep_ci(roc_auc, y, prob, g, seed_off=61)
            ap_cl = perrep_ci(auprc, y, prob, g, seed_off=62)
            br_cl = perrep_ci(brier, y, prob, g, seed_off=63)
            ci_cl = perrep_ci(lambda a, b: calib_logodds(a, b)[0], y, prob, g,
                              n=N_BOOT_SUB, seed_off=64)
            cs_cl = perrep_ci(lambda a, b: calib_logodds(a, b)[1], y, prob, g,
                              n=N_BOOT_SUB, seed_off=66)
            met_rows.append({
                "drug": drug, "role": role, "model": model, "n": int(len(y)),
                "n_patients": int(np.unique(g).size), "prevalence": round(float(y.mean()), 3),
                "auroc_mean": round(summ["auroc"][0], 3), "auroc_min": round(summ["auroc"][1], 3),
                "auroc_max": round(summ["auroc"][2], 3),
                "auroc_boot_lo": round(au_cl[0], 3), "auroc_boot_hi": round(au_cl[1], 3),
                "auprc_mean": round(summ["auprc"][0], 3),
                "auprc_boot_lo": round(ap_cl[0], 3), "auprc_boot_hi": round(ap_cl[1], 3),
                "brier_mean": round(summ["brier"][0], 4), "brier_min": round(summ["brier"][1], 4),
                "brier_max": round(summ["brier"][2], 4),
                "brier_boot_lo": round(br_cl[0], 4), "brier_boot_hi": round(br_cl[1], 4),
                "calib_intercept_mean": round(summ["calib_intercept"][0], 3),
                "calib_intercept_lo": round(ci_cl[0], 3), "calib_intercept_hi": round(ci_cl[1], 3),
                "calib_slope_mean": round(summ["calib_slope"][0], 3),
                "calib_slope_lo": round(cs_cl[0], 3), "calib_slope_hi": round(cs_cl[1], 3)})

            # ---- OOF training-fold prevalence baseline + Brier skill, PER MODEL ----
            prev_briers = []
            for fold in folds:
                pv = np.full(len(y), np.nan)
                for test_pats in fold:
                    te = np.isin(g, test_pats)
                    pv[te] = y[~te].mean()
                prev_briers.append(brier(y, pv))
            base_brier = float(np.mean(prev_briers))
            base_rows.append({
                "drug": drug, "role": role, "model": model, "n": int(len(y)),
                "prevalence_whole_cohort": round(float(y.mean()), 4),
                "brier_model_mean": round(summ["brier"][0], 4),
                "brier_oof_prevalence_mean": round(base_brier, 4),
                "brier_skill_vs_oof_prevalence": round(1.0 - summ["brier"][0] / base_brier, 4)})

            # ---- below-65 subgroup, per partition ----
            if smask.sum() >= 50 and len(np.unique(y[smask])) == 2:
                s_au = perrep_point(roc_auc, y, prob, mask=smask)
                s_au_ci = perrep_ci(roc_auc, y, prob, g, mask=smask, n=N_BOOT_SUB, seed_off=81)
                s_br = perrep_point(brier, y, prob, mask=smask)
                s_br_ci = perrep_ci(brier, y, prob, g, mask=smask, n=N_BOOT_SUB, seed_off=83)
                s_in = perrep_point(lambda a, b: calib_logodds(a, b)[0], y, prob, mask=smask)
                s_in_ci = perrep_ci(lambda a, b: calib_logodds(a, b)[0], y, prob, g,
                                    mask=smask, n=N_BOOT_SUB, seed_off=85)
                s_sl = perrep_point(lambda a, b: calib_logodds(a, b)[1], y, prob, mask=smask)
                s_sl_ci = perrep_ci(lambda a, b: calib_logodds(a, b)[1], y, prob, g,
                                    mask=smask, n=N_BOOT_SUB, seed_off=87)
                sub_rows.append({
                    "drug": drug, "model": model, "subgroup": "baseline_map_lt65",
                    "n": int(smask.sum()), "n_patients": int(np.unique(g[smask]).size),
                    "observed_attainment": round(float(y[smask].mean()), 3),
                    "auroc": round(s_au, 3), "auroc_lo": round(s_au_ci[0], 3),
                    "auroc_hi": round(s_au_ci[1], 3),
                    "brier": round(s_br, 4), "brier_lo": round(s_br_ci[0], 4),
                    "brier_hi": round(s_br_ci[1], 4),
                    "calib_intercept": round(s_in, 3), "calib_intercept_lo": round(s_in_ci[0], 3),
                    "calib_intercept_hi": round(s_in_ci[1], 3),
                    "calib_slope": round(s_sl, 3), "calib_slope_lo": round(s_sl_ci[0], 3),
                    "calib_slope_hi": round(s_sl_ci[1], 3)})

            # ---- descriptive operating points, per-partition mean ----
            for scope, msk in [("overall", None), ("baseline_lt65", smask)]:
                if msk is not None and not (msk.sum() >= 50 and len(np.unique(y[msk])) == 2):
                    continue
                for thr in OP_GRID:
                    sens_r, spec_r, ppv_r, npv_r = [], [], [], []
                    for r in range(R):
                        yy = y if msk is None else y[msk]
                        pp = prob[r] if msk is None else prob[r][msk]
                        yh = (pp >= thr).astype(int)
                        tp = int(((yh == 1) & (yy == 1)).sum()); fp = int(((yh == 1) & (yy == 0)).sum())
                        fn_ = int(((yh == 0) & (yy == 1)).sum()); tn = int(((yh == 0) & (yy == 0)).sum())
                        if tp + fn_: sens_r.append(tp / (tp + fn_))
                        if tn + fp: spec_r.append(tn / (tn + fp))
                        if tp + fp: ppv_r.append(tp / (tp + fp))
                        if tn + fn_: npv_r.append(tn / (tn + fn_))
                    op_rows.append({
                        "drug": drug, "model": model, "scope": scope, "threshold": thr,
                        "n": int(len(y) if msk is None else msk.sum()),
                        "sensitivity": round(float(np.mean(sens_r)), 3) if sens_r else np.nan,
                        "specificity": round(float(np.mean(spec_r)), 3) if spec_r else np.nan,
                        "ppv": round(float(np.mean(ppv_r)), 3) if ppv_r else np.nan,
                        "npv": round(float(np.mean(npv_r)), 3) if npv_r else np.nan})

            # ---- pooled reliability curve (stability summary), xgb only for figure ----
            if model == "xgb":
                pooled = prob.mean(axis=0)
                _calib_curve(curve_rows, drug, "overall", y, pooled, g)
                if smask.sum() >= 50 and len(np.unique(y[smask])) == 2:
                    _calib_curve(curve_rows, drug, "baseline_lt65", y[smask],
                                 pooled[smask], g[smask])

    pd.DataFrame(per_rows).to_csv(OUT / "classifier_perpartition.csv", index=False)
    pd.DataFrame(met_rows).to_csv(OUT / "classifier_metrics.csv", index=False)
    pd.DataFrame(sub_rows).to_csv(OUT / "classifier_subgroup.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT / "calibration_curve.csv", index=False)
    pd.DataFrame(op_rows).to_csv(OUT / "operating_points.csv", index=False)
    pd.DataFrame(base_rows).to_csv(OUT / "prob_baseline.csv", index=False)
    log.info("classifier tables written")


# ------------------------------------------------------------------- follow-up strata
def followup_strata(d):
    """Follow-up-treatment strata MAE, per partition. Primary candidate model (XGBoost +
    history). Three overlapping strata within 5 min: any additional vasopressor
    (union), a different-drug vasopressor, and a same-drug re-dose."""
    rows = []
    for drug in DRUGS:
        sub = sub_for(d, drug)
        if len(sub) < 300:
            continue
        role = "exploratory" if drug in EXPLORATORY else "primary"
        y = sub["delta_map_5min"].values.astype(float)
        g = sub["subjectid"].values
        oof = oof_reg(drug, "base_hist", "xgb")
        other = sub["fu_other_vaso_5m"].values.astype(float) > 0
        same = sub["fu_same_redose_5m"].values.astype(float) > 0
        # the first stratum is the union of the other two; the strata overlap and are
        # not a partition of the cohort
        flags = {"any additional vasopressor within 5 min (incl. same-drug re-dose)": other | same,
                 "different-drug vasopressor within 5 min": other,
                 "same-drug re-dose within 5 min": same}
        for name, present_mask in flags.items():
            for present, msk in [(0, ~present_mask), (1, present_mask)]:
                if msk.sum() < 50:
                    continue
                m = perrep_point(mae, y, oof, mask=msk)
                lo, hi = perrep_ci(mae, y, oof, g, mask=msk, n=N_BOOT_SUB, seed_off=91)
                rows.append({"drug": drug, "role": role, "intervention": name,
                             "present": present, "n": int(msk.sum()),
                             "n_patients": int(np.unique(g[msk]).size),
                             "mae": round(m, 3), "lo": round(lo, 3), "hi": round(hi, 3),
                             "observed_delta_mean": round(float(y[msk].mean()), 2),
                             "baseline_map_mean": round(float(sub["baseline_map"].values[msk].mean()), 1)})
    pd.DataFrame(rows).to_csv(OUT / "followup_strata.csv", index=False)
    log.info("follow-up strata written")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list: regression,baseline,did,classifier,followup")
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)
    d = load_events()
    log.info(f"event master: {len(d)} rows")
    if not only or "regression" in only: regression_tables(d)
    if not only or "baseline" in only: baseline_r2_table(d)
    if not only or "did" in only: did_table(d)
    if not only or "classifier" in only: classifier_tables(d)
    if not only or "followup" in only: followup_strata(d)
    log.info("per-partition scoring done")


if __name__ == "__main__":
    main()
