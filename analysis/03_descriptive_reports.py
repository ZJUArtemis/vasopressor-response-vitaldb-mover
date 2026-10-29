#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Descriptive tables that need only the event master (no model fitting), plus the
baseline-only R^2 reference computed from the baseline-model out-of-fold predictions.

  - cohort flow from MAP-linked administrations to the primary bolus-class cohort
  - follow-up treatment within the outcome window (other vasopressor / same-drug re-dose / union)
  - feature availability and missingness, distinguishing structural "no prior dose" absence
  - coverage and age of the most recent recorded infusion rate
  - baseline-MAP-only R^2 versus the full model

Outputs (results/): cohort_flow_full.csv, followup_union.csv, followup_strata_pooled.csv,
missingness.csv, running_rate_desc.csv, baseline_only_r2.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # repository root
DATA = ROOT / "data"                          # event_master.parquet (not distributed)
RES = ROOT / "results"
EV = DATA / "event_master.parquet"
OUT = RES
OUT.mkdir(parents=True, exist_ok=True)
STUDY = ["phenylephrine", "ephedrine", "vasopressin", "norepinephrine"]
PRIMARY = ["phenylephrine", "ephedrine", "vasopressin"]

d = pd.read_parquet(EV)
study = d[d.drug.isin(STUDY)].copy()


def flow():
    rows = []
    tot = len(study)
    rows.append({'step': 'map_linked_study_drug_administrations', 'n_events': tot,
                 'n_patients': study.subjectid.nunique()})
    for cls, lab in [('iv_bolus', 'iv_bolus'),
                     ('infusion_start_or_bag_change', 'infusion_start_or_bag_change'),
                     ('infusion_resume', 'infusion_resume'),
                     ('non_iv_excluded', 'non_iv'),
                     ('uncertain_unit', 'uncertain_unit'),
                     ('rate_documented_as_given', 'rate_documented_as_given')]:
        s = study[study.event_class_v12 == cls]
        rows.append({'step': f'class::{lab}', 'n_events': len(s), 'n_patients': s.subjectid.nunique()})
    bolus = study[study.event_class_v12 == 'iv_bolus']
    rows.append({'step': 'bolus_all_study_drugs', 'n_events': len(bolus), 'n_patients': bolus.subjectid.nunique()})
    prim = bolus[bolus.drug.isin(PRIMARY)]
    rows.append({'step': 'bolus_three_primary_drugs', 'n_events': len(prim), 'n_patients': prim.subjectid.nunique()})
    nore = bolus[bolus.drug == 'norepinephrine']
    rows.append({'step': 'bolus_norepinephrine_exploratory', 'n_events': len(nore), 'n_patients': nore.subjectid.nunique()})
    epi = study[(study.event_class_v12 == 'iv_bolus') & (~study.drug.isin(STUDY))]
    rows.append({'step': 'reconcile_check_bolus=prim+nore', 'n_events': len(prim) + len(nore), 'n_patients': -1})
    pd.DataFrame(rows).to_csv(OUT / "cohort_flow.csv", index=False)
    return pd.DataFrame(rows)


def followup_union():
    bolus = study[(study.event_class_v12 == 'iv_bolus') & (study.drug.isin(PRIMARY))]
    rows = []
    for drug in PRIMARY:
        s = bolus[bolus.drug == drug]
        other = s.fu_other_vaso_5m.fillna(0) > 0
        redose = s.fu_same_redose_5m.fillna(0) > 0
        union = (other | redose)
        rows.append({'drug': drug, 'n': len(s), 'n_patients': s.subjectid.nunique(),
                     'other_vaso_5m_pct': round(100 * other.mean(), 2),
                     'same_redose_5m_pct': round(100 * redose.mean(), 2),
                     'union_5m_pct': round(100 * union.mean(), 2)})
    pd.DataFrame(rows).to_csv(OUT / "followup_union.csv", index=False)
    return pd.DataFrame(rows)


# features whose absence is structural ("never dosed") vs genuinely missing
HIST_STRUCTURAL = {'hist_min_since_same', 'hist_min_since_any_vaso', 'hist_running_rate_same',
                   'hist_running_rate_unit', 'hist_running_rate_age_min'}
BASE_NUM = ['age', 'sex_male', 'bmi', 'asa', 'dose_rate_at_event', 'baseline_map',
            'ctx_map_mean', 'ctx_map_min', 'ctx_map_last', 'ctx_map_trend', 'ctx_map_cv',
            'ctx_time_below_65_min']
HIST_NUM = ['hist_same_n_5m', 'hist_any_vaso_n_5m', 'hist_same_n_10m', 'hist_any_vaso_n_10m',
            'hist_same_n_30m', 'hist_any_vaso_n_30m', 'hist_same_first_dose', 'hist_any_first_vaso',
            'hist_min_since_same', 'hist_min_since_any_vaso', 'hist_running_rate_same',
            'hist_rate_changes_30m', 'hist_same_bolus_dose_10m', 'hist_same_bolus_dose_30m']


def missingness():
    bolus = study[(study.event_class_v12 == 'iv_bolus') & (study.drug.isin(PRIMARY))]
    rows = []
    for drug in PRIMARY:
        s = bolus[bolus.drug == drug]
        n = len(s)
        for block, cols in [('base', BASE_NUM), ('history', HIST_NUM)]:
            for c in cols:
                if c not in s.columns:
                    continue
                miss = s[c].isna().mean()
                structural = c in HIST_STRUCTURAL
                # for gap/rate features, "no prior dose" is structural absence:
                never = float((s['hist_any_vaso_n_30m'].fillna(0) == 0).mean()) if structural else np.nan
                rows.append({'drug': drug, 'block': block, 'feature': c, 'n': n,
                             'missing_pct': round(100 * float(miss), 2),
                             'structural_absence': structural,
                             'no_prior_vaso_30m_pct': round(100 * never, 2) if structural else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "missingness.csv", index=False)
    return pd.DataFrame(rows)


def running_rate():
    bolus = study[(study.event_class_v12 == 'iv_bolus') & (study.drug.isin(PRIMARY))]
    rows = []
    for drug in PRIMARY:
        s = bolus[bolus.drug == drug]
        has = s.hist_running_rate_age_min.notna()
        age = s.loc[has, 'hist_running_rate_age_min']
        rows.append({'drug': drug, 'n': len(s), 'pct_with_recent_rate': round(100 * float(has.mean()), 2),
                     'rate_age_median_min': round(float(age.median()), 1) if has.any() else np.nan,
                     'rate_age_p25_min': round(float(age.quantile(.25)), 1) if has.any() else np.nan,
                     'rate_age_p75_min': round(float(age.quantile(.75)), 1) if has.any() else np.nan,
                     'n_distinct_units': int(s.loc[has, 'hist_running_rate_unit'].nunique())})
    pd.DataFrame(rows).to_csv(OUT / "running_rate_desc.csv", index=False)
    return pd.DataFrame(rows)


def baseline_only_r2():
    """Compare baseline-MAP-only R^2 with the full XGB+history R^2 on identical events.
    Uses the retrained OOF arrays if present; skips gracefully otherwise."""
    rows = []
    for drug in PRIMARY + ['norepinephrine']:
        base_f = OUT / f"oof_baseline_ridge_{drug}.npy"
        full_f = OUT / f"oof_base_hist_xgb_{drug}.npy"
        if not (base_f.exists() and full_f.exists()):
            continue
        s = study[(study.event_class_v12 == 'iv_bolus') & (study.drug == drug)]
        y = s.delta_map_5min.values.astype(float)
        pb = np.load(base_f).mean(axis=0); pf = np.load(full_f).mean(axis=0)
        sst = np.sum((y - y.mean()) ** 2)
        r2b = 1 - np.sum((y - pb) ** 2) / sst
        r2f = 1 - np.sum((y - pf) ** 2) / sst
        rows.append({'drug': drug, 'n': len(y), 'r2_baseline_only': round(float(r2b), 4),
                     'r2_full_xgb_hist': round(float(r2f), 4),
                     'r2_increment_over_baseline': round(float(r2f - r2b), 4)})
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "baseline_only_r2.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== cohort flow ==="); print(flow().to_string(index=False))
    print("\n=== follow-up union ==="); print(followup_union().to_string(index=False))
    print("\n=== running rate ==="); print(running_rate().to_string(index=False))
    print("\n=== missingness (head) ==="); print(missingness().head(12).to_string(index=False))
    r2 = baseline_only_r2()
    print("\n=== baseline-only R2 ==="); print(r2.to_string(index=False) if len(r2) else "(OOF not ready yet)")
