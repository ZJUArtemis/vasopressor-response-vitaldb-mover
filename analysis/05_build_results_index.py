#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the single results index from which every load-bearing number in the manuscript
is drawn.

Folds the per-partition scoring outputs and the regression comparison into
results/results_index.csv, keyed by (drug, estimand, features, model, comparator).
"""
import sys
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
V16 = RES
V14 = RES
OUT = RES / "results_index.csv"

COMP_FEATURES = {
    'history_effect_on_xgb': 'xgb',
    'history_effect_on_ridge': 'ridge',
    'algorithm_effect_base': 'xgb_vs_ridge_base',
    'algorithm_effect_with_hist': 'xgb_vs_ridge_hist',
    'best_candidate_vs_baseline': 'xgb_hist',
}


def need(base, p):
    fp = base / p
    if not fp.exists():
        print(f"FATAL: missing input {fp}"); sys.exit(2)
    return pd.read_csv(fp)


def main():
    delta = need(V14, "delta_mae.csv")
    comp = need(V14, "model_comparison.csv")
    qual = need(V16, "regression_quality.csv")
    br = need(V16, "baseline_r2.csv")
    did = need(V16, "did.csv")
    strata = need(V16, "map_strata.csv")
    clf = need(V16, "classifier_metrics.csv")
    sg = need(V16, "classifier_subgroup.csv")
    pb = need(V16, "prob_baseline.csv")

    rows = []

    def add(drug, estimand, estimate, lo='', hi='', features='', model='',
            comparator='', n='', n_patients='', source='', role='primary'):
        rows.append({'drug': drug, 'estimand': estimand, 'features': features,
                     'model': model, 'comparator': comparator, 'estimate': estimate,
                     'ci_lo': lo, 'ci_hi': hi, 'n': n, 'n_patients': n_patients,
                     'source': source, 'role': role})

    # 1) paired dMAE comparisons (regression comparison)
    for _, r in delta.iterrows():
        feat = COMP_FEATURES.get(r['comparison'], r['comparison'])
        add(r['drug'], 'dMAE', r['dmae'], r['ci_lo'], r['ci_hi'], feat, '',
            r['comparison'], r['n'], r['n_patients'], 'delta_mae.csv', r['role'])

    # 2) MAE per model x feature set
    for _, r in comp.iterrows():
        add(r['drug'], 'MAE', r['mae_mean'], '', '', r['features'], r['model'], '',
            r['n'], r['n_patients'], 'model_comparison.csv', r['role'])

    # 3) regression quality of primary candidate (xgb + history), per-repeat
    for _, r in qual.iterrows():
        role = r['role']
        add(r['drug'], 'r2', r['r2_mean'], r['r2_boot_lo'], r['r2_boot_hi'],
            'xgb_hist', 'xgb', '', r['n'], r['n_patients'], 'regression_quality.csv', role)
        add(r['drug'], 'mae_over_sd', r['mae_over_sd_mean'], '', '', 'xgb_hist', 'xgb', '',
            r['n'], r['n_patients'], 'regression_quality.csv', role)
        add(r['drug'], 'rmse', r['rmse_mean'], '', '', 'xgb_hist', 'xgb', '',
            r['n'], r['n_patients'], 'regression_quality.csv', role)
        add(r['drug'], 'mae', r['mae_mean'], '', '', 'xgb_hist', 'xgb', '',
            r['n'], r['n_patients'], 'regression_quality.csv', role)
        add(r['drug'], 'outcome_sd', r['outcome_sd'], '', '', 'xgb_hist', 'xgb', '',
            r['n'], r['n_patients'], 'regression_quality.csv', role)
        add(r['drug'], 'reg_calib_slope', r['calib_slope_mean'], r['calib_slope_boot_lo'],
            r['calib_slope_boot_hi'], 'xgb_hist', 'xgb', '', r['n'], r['n_patients'],
            'regression_quality.csv', role)
        for thr in (5, 10, 15):
            add(r['drug'], f'pct_err_gt_{thr}', r[f'pct_err_gt_{thr}_mean'],
                r[f'pct_err_gt_{thr}_lo'], r[f'pct_err_gt_{thr}_hi'], 'xgb_hist', 'xgb', '',
                r['n'], r['n_patients'], 'regression_quality.csv', role)

    # 4) baseline-only r2 + increment
    for _, r in br.iterrows():
        role = 'exploratory' if r['drug'] == 'norepinephrine' else 'primary'
        add(r['drug'], 'r2_baseline_only', r['r2_baseline_only_mean'], '', '',
            'baseline_map', 'baseline', '', r['n'], '', 'baseline_r2.csv', role)
        add(r['drug'], 'r2_increment', r['r2_increment'], '', '', 'xgb_hist', 'xgb', '',
            r['n'], '', 'baseline_r2.csv', role)

    # 5) difference-in-differences (per-repeat; equals difference of history effects)
    for _, r in did.iterrows():
        add(r['drug'], 'did_history', r['did'], r['ci_lo'], r['ci_hi'], 'xgb_vs_ridge_hist',
            '', 'history_did', r['n'], r['n_patients'], 'did.csv', r['role'])

    # 6) MAP strata MAE (per-repeat)
    for _, r in strata.iterrows():
        add(r['drug'], 'strata_mae', r['mae'], r['lo'], r['hi'], 'xgb_hist', 'xgb',
            r['stratum'], r['n'], r['n_patients'], 'map_strata.csv', r['role'])
        add(r['drug'], 'strata_obs_delta', r['observed_delta_mean'], '', '', 'xgb_hist', 'xgb',
            r['stratum'], r['n'], r['n_patients'], 'map_strata.csv', r['role'])

    # 7) whole-cohort classifier metrics (logit + xgb), per-repeat
    for _, r in clf.iterrows():
        model = f"clf_{r['model']}"
        add(r['drug'], 'auroc', r['auroc_mean'], r['auroc_boot_lo'], r['auroc_boot_hi'],
            'whole', model, '', r['n'], r['n_patients'], 'classifier_metrics.csv', r['role'])
        add(r['drug'], 'auprc', r['auprc_mean'], r['auprc_boot_lo'], r['auprc_boot_hi'],
            'whole', model, '', r['n'], r['n_patients'], 'classifier_metrics.csv', r['role'])
        add(r['drug'], 'brier', r['brier_mean'], r['brier_boot_lo'], r['brier_boot_hi'],
            'whole', model, '', r['n'], r['n_patients'], 'classifier_metrics.csv', r['role'])
        add(r['drug'], 'prevalence', r['prevalence'], '', '', 'whole', model, '', r['n'],
            r['n_patients'], 'classifier_metrics.csv', r['role'])
        add(r['drug'], 'calib_intercept', r['calib_intercept_mean'], r['calib_intercept_lo'],
            r['calib_intercept_hi'], 'whole', model, '', r['n'], r['n_patients'],
            'classifier_metrics.csv', r['role'])
        add(r['drug'], 'calib_slope', r['calib_slope_mean'], r['calib_slope_lo'],
            r['calib_slope_hi'], 'whole', model, '', r['n'], r['n_patients'],
            'classifier_metrics.csv', r['role'])

    # 8) below-65 subgroup classifier (logit + xgb), per-repeat
    for _, r in sg.iterrows():
        role = 'exploratory' if r['drug'] == 'norepinephrine' else 'primary'
        model = f"clf_{r['model']}"
        add(r['drug'], 'auroc', r['auroc'], r['auroc_lo'], r['auroc_hi'], 'below65', model,
            '', r['n'], r['n_patients'], 'classifier_subgroup.csv', role)
        add(r['drug'], 'brier', r['brier'], r['brier_lo'], r['brier_hi'], 'below65', model,
            '', r['n'], r['n_patients'], 'classifier_subgroup.csv', role)
        add(r['drug'], 'calib_intercept', r['calib_intercept'], r['calib_intercept_lo'],
            r['calib_intercept_hi'], 'below65', model, '', r['n'], r['n_patients'],
            'classifier_subgroup.csv', role)
        add(r['drug'], 'calib_slope', r['calib_slope'], r['calib_slope_lo'], r['calib_slope_hi'],
            'below65', model, '', r['n'], r['n_patients'], 'classifier_subgroup.csv', role)

    # 9) Brier skill vs OOF prevalence (per model)
    for _, r in pb.iterrows():
        model = f"clf_{r['model']}"
        add(r['drug'], 'brier_skill', r['brier_skill_vs_oof_prevalence'], '', '', 'whole',
            model, '', r['n'], '', 'prob_baseline.csv', r['role'])

    reg = pd.DataFrame(rows, columns=[
        'drug', 'estimand', 'features', 'model', 'comparator',
        'estimate', 'ci_lo', 'ci_hi', 'n', 'n_patients', 'source', 'role'])
    reg.to_csv(OUT, index=False)
    print(f"wrote {OUT} with {len(reg)} rows")
    print("drugs:", sorted(reg.drug.unique()))
    print("estimands:", sorted(reg.estimand.unique()))


if __name__ == "__main__":
    main()
