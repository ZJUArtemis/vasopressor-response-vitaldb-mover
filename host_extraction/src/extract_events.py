#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Host-side event extraction from the MOVER medication administration record (MAR)
and arterial-pressure stream. No modelling.

For every documented administration of a study drug inside a valid surgery window
with an eligible pre- and post-dose arterial-MAP window, the script
  A. assigns a mutually exclusive exposure class from the MAR action string, dose
     unit and route (rule fixed before analysis; see config/mover_mar_actions.yaml);
  B. computes the pre-dose treatment-history features strictly from records
     timestamped before t0 (same-drug and any-vasopressor counts, cumulative dose,
     time since last dose, most recent recorded infusion rate);
  C. flags further treatment within 5/10 min after t0 (another vasopressor, a
     same-drug re-dose, a new infusion, a rate change);
  D. optionally reconciles the event identities and patient set against a previous
     extraction (--previous_events).

Outputs (all in ./outputs/; de-identified, no free text, no dates)
  v12_event_master.parquet         one row per MAP-linked administration (the table the
                                   local analysis consumes as data/event_master.parquet)
  v12_action_census.csv            MAR action x dose unit x route, per drug, with class and counts
  v12_event_class_counts.csv       per-drug class counts
  v12_history_availability.csv     fraction of events with >=1 prior record in 5/10/30 min
  v12_history_features_summary.csv distribution of each history feature per drug
  v12_followup_census.csv          fraction and timing of each post-dose intervention
  v12_infusion_reconstruction.csv  how well a running rate can be reconstructed at t0
  v12_window_quality.csv           MAP sampling density and NIBP/ABP mixing per event
  v12_summary.json

Output file names and the event_class_v12 / etype_v7 column names are kept because the
downstream analysis reads them by name. CPU only. Reads forward-only. Writes nothing
outside ./outputs/.
"""
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mover_io import (FEATURES_COMMON, TARGET, GROUP, MAP_MIN, MAP_MAX,
                     setup_log, load_cfg, parse_time, load_raw, window_stats)

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "outputs"

HIST_WINDOWS_MIN = [5, 10, 30]
FOLLOW_WINDOWS_MIN = [5, 10]
RATE_UNITS = {'mcg/min', 'mcg/kg/min', 'Units/min', 'mg/hr', 'mcg/hr', 'Units/hr', 'mg/min'}
BOLUS_UNITS = {'mcg', 'mg', 'Units', 'units'}

# exposure classes — rule fixed before analysis, applied to the action string + dose unit.
# "uncertain" is a real class, not a fallback: it is what the manuscript must
# report as 'not classifiable from the record'.
def classify(action, unit, route):
    a = str(action); u = str(unit) if unit is not None else ''; r = str(route).lower() if route is not None else ''
    non_iv = any(k in r for k in ['infiltrat', 'topical', 'subcut', 'intraderm', 'nebul', 'irrig',
                                   'oral', 'buccal', 'eye', 'naris', 'nasal', 'epidural', 'intrathecal',
                                   'perineural', 'cervical', 'vaginal', 'other', 'soak'])
    if non_iv:
        return 'non_iv_excluded'
    if a in ('Given', 'Bolus', 'Bolus From Bag', 'Given by Other', 'Given By Pt', 'Given in incremental doses'):
        if u in BOLUS_UNITS:
            return 'iv_bolus'
        if u in RATE_UNITS:
            return 'rate_documented_as_given'      # a rate recorded under a bolus action: ambiguous
        return 'uncertain_unit'
    if a == 'New Bag':
        return 'infusion_start_or_bag_change'      # cannot be split without order context
    if a in ('Restarted', 'ED Continuing Infusion', 'IV Resume', 'IV Resume Dual Sign', 'Restarted Dual Sign'):
        return 'infusion_resume'
    if a in ('Rate Change', 'Rate Change Dual Sign'):
        return 'rate_change'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--max_cases', type=int, default=0)
    ap.add_argument('--previous_events', default=None,
                    help='optional path to an event table from a previous extraction, for reconciliation')
    args = ap.parse_args()
    setup_log("extract_events")
    cfg, actions = load_cfg(args.config)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.info("=== MOVER event extraction (no modelling) ===")

    # ------------------------------------------------------------------
    # Load with the shared loader, but ALSO keep rate-change rows (the loader keeps them with
    # _etype='rate_increase'; they are dropped from START events but are needed
    # here for history and follow-up features).
    # ------------------------------------------------------------------
    med_by, map_by, win, pat, ccol, cases = load_raw(cfg, actions, max_cases=args.max_cases)
    logging.info(f"cases with documented candidates: {len(cases)}")
    ocol = cfg.get('med_order_id_col') or None
    rcol = cfg.get('med_route_col') or None
    dcol, ucol = cfg['med_dose_col'], cfg['med_dose_unit_col']

    # ------------------------------------------------------------------
    # A. Action census with exposure class
    # ------------------------------------------------------------------
    rows = []
    for case in cases:
        mc = med_by[case]
        for _, ev in mc.iterrows():
            rows.append({'drug': ev['_drug'], 'mar_action': str(ev['_action']), 'etype_v7': ev['_etype'],
                         'dose_unit': ev.get(ucol), 'route': (ev.get(rcol) if rcol else None),
                         'event_class_v12': classify(ev['_action'], ev.get(ucol), ev.get(rcol) if rcol else None)})
    cen = pd.DataFrame(rows)
    cen_out = (cen.groupby(['drug', 'mar_action', 'etype_v7', 'dose_unit', 'route', 'event_class_v12'], dropna=False)
               .size().reset_index(name='n_rows').sort_values(['drug', 'n_rows'], ascending=[True, False]))
    cen_out.to_csv(OUT / "v12_action_census.csv", index=False)
    cls = cen.groupby(['drug', 'event_class_v12']).size().unstack(fill_value=0)
    cls.to_csv(OUT / "v12_event_class_counts.csv")
    logging.info(f"event classes:\n{cls}")

    # ------------------------------------------------------------------
    # Build event master: every START-type (administration) row in a valid
    # OR window with an eligible MAP window, plus
    # history + follow-up features computed from the full per-case MAR.
    # ------------------------------------------------------------------
    master = []
    for case in cases:
        if case not in map_by or map_by[case] is None or case not in win:
            continue
        mc = med_by[case].copy()
        w = win[case]
        mapcase = map_by[case]
        mc['_dose'] = pd.to_numeric(mc[dcol], errors='coerce')
        mc['_unit'] = mc[ucol].astype(str)
        mc['_class'] = [classify(a, u, r) for a, u, r in
                        zip(mc['_action'], mc[ucol], (mc[rcol] if rcol else [None] * len(mc)))]
        prow = pat.loc[case] if (pat is not None and case in pat.index) else None
        def pget(key):
            c = cfg.get(key)
            return prow[c] if (prow is not None and c in getattr(prow, 'index', [])) else None
        subj = pget('mrn_col') if pget('mrn_col') is not None else case
        for _, ev in mc.iterrows():
            if ev['_etype'] != 'start':
                continue
            t0 = ev['_t']; drug = ev['_drug']
            if not (w['_in'] <= t0 <= w['_out']):
                continue
            st = window_stats(mapcase, t0, offset_min=0)
            if st is None:
                continue
            rec = {'caseid': case, 'subjectid': subj, 'drug': drug,
                   'event_id': f"{case}:{drug}:{int(t0.timestamp())}",
                   't0_minutes_from_or_in': round((t0 - w['_in']).total_seconds() / 60, 1),
                   'mar_action': str(ev['_action']), 'event_class_v12': ev['_class'],
                   'dose_rate_at_event': ev['_dose'], 'dose_unit': ev.get(ucol),
                   'route': ev.get(rcol) if rcol else None,
                   'age': pd.to_numeric(pget('age_col'), errors='coerce') if pget('age_col') is not None else np.nan,
                   'sex_male': 1 if (pget('sex_col') is not None and str(pget('sex_col')).upper().startswith('M')) else 0,
                   'asa': pd.to_numeric(pget('asa_col'), errors='coerce') if pget('asa_col') is not None else np.nan,
                   'bmi': np.nan}
            h, wt = pget('height_col'), pget('weight_col')
            if h is not None and wt is not None:
                h = pd.to_numeric(h, errors='coerce'); wt = pd.to_numeric(wt, errors='coerce')
                if np.isfinite(h) and np.isfinite(wt) and h > 0:
                    rec['bmi'] = round(float(wt / ((h / 100) ** 2)), 1)
            rec.update(st)

            # ---- pre-dose history: STRICTLY before t0 ----
            before = mc[mc['_t'] < t0]
            same_b = before[before['_drug'] == drug]
            for wmin in HIST_WINDOWS_MIN:
                lo = t0 - pd.Timedelta(minutes=wmin)
                sw = same_b[same_b['_t'] >= lo]; aw = before[before['_t'] >= lo]
                rec[f'hist_same_n_{wmin}m'] = int((sw['_etype'] == 'start').sum())
                rec[f'hist_any_vaso_n_{wmin}m'] = int((aw['_etype'] == 'start').sum())
                # cumulative same-drug BOLUS dose only, same unit as the index dose, else NaN
                sb = sw[(sw['_class'] == 'iv_bolus') & (sw['_unit'] == str(ev.get(ucol)))]
                rec[f'hist_same_bolus_dose_{wmin}m'] = float(sb['_dose'].sum()) if len(sb) else 0.0
                rec[f'hist_same_bolus_dose_{wmin}m_unit_match'] = int(len(sb) == int((sw['_class'] == 'iv_bolus').sum()))
            rec['hist_same_first_dose'] = int(len(same_b[same_b['_etype'] == 'start']) == 0)
            rec['hist_any_first_vaso'] = int(len(before[before['_etype'] == 'start']) == 0)
            ls = same_b[same_b['_etype'] == 'start']['_t']
            la = before[before['_etype'] == 'start']['_t']
            rec['hist_min_since_same'] = round((t0 - ls.max()).total_seconds() / 60, 1) if len(ls) else np.nan
            rec['hist_min_since_any_vaso'] = round((t0 - la.max()).total_seconds() / 60, 1) if len(la) else np.nan
            # running infusion rate at t0: last rate-unit row of the same drug before t0
            rr = same_b[same_b['_unit'].isin(RATE_UNITS)]
            rec['hist_running_rate_same'] = float(rr['_dose'].iloc[-1]) if len(rr) else 0.0
            rec['hist_running_rate_unit'] = rr['_unit'].iloc[-1] if len(rr) else None
            rec['hist_running_rate_age_min'] = round((t0 - rr['_t'].iloc[-1]).total_seconds() / 60, 1) if len(rr) else np.nan
            rec['hist_rate_changes_30m'] = int(((before['_t'] >= t0 - pd.Timedelta(minutes=30)) &
                                                (before['_etype'] == 'rate_increase')).sum())

            # ---- follow-up: AFTER t0 (interpretation only; never a feature) ----
            after = mc[mc['_t'] > t0]
            for wmin in FOLLOW_WINDOWS_MIN:
                hi = t0 + pd.Timedelta(minutes=wmin)
                aw = after[after['_t'] <= hi]
                sd = aw[(aw['_drug'] == drug) & (aw['_etype'] == 'start') & (aw['_t'] > t0 + pd.Timedelta(seconds=30))]
                od = aw[(aw['_drug'] != drug) & (aw['_etype'] == 'start')]
                nb = aw[(aw['_drug'] != drug) & (aw['_class'] == 'infusion_start_or_bag_change')]
                rc = aw[aw['_etype'] == 'rate_increase']
                rec[f'fu_same_redose_{wmin}m'] = int(len(sd) > 0)
                rec[f'fu_same_redose_{wmin}m_first_min'] = round((sd['_t'].min() - t0).total_seconds() / 60, 1) if len(sd) else np.nan
                rec[f'fu_other_vaso_{wmin}m'] = int(len(od) > 0)
                rec[f'fu_other_vaso_{wmin}m_first_min'] = round((od['_t'].min() - t0).total_seconds() / 60, 1) if len(od) else np.nan
                rec[f'fu_new_infusion_other_{wmin}m'] = int(len(nb) > 0)
                rec[f'fu_rate_change_any_{wmin}m'] = int(len(rc) > 0)
            master.append(rec)
    ev = pd.DataFrame(master)
    ev.to_parquet(OUT / "v12_event_master.parquet", index=False)
    logging.info(f"event master: {len(ev)} events, {ev.subjectid.nunique()} patients")

    # ------------------------------------------------------------------
    # B. History availability & feature distributions
    # ------------------------------------------------------------------
    ha, hs = [], []
    for drug, d in ev.groupby('drug'):
        row = {'drug': drug, 'n_events': len(d), 'n_patients': d.subjectid.nunique(),
               'frac_first_same_dose': round(float(d.hist_same_first_dose.mean()), 3),
               'frac_first_any_vaso': round(float(d.hist_any_first_vaso.mean()), 3),
               'frac_running_rate_present': round(float((d.hist_running_rate_same > 0).mean()), 3)}
        for wmin in HIST_WINDOWS_MIN:
            row[f'frac_prior_same_{wmin}m'] = round(float((d[f'hist_same_n_{wmin}m'] > 0).mean()), 3)
            row[f'frac_prior_any_{wmin}m'] = round(float((d[f'hist_any_vaso_n_{wmin}m'] > 0).mean()), 3)
            row[f'frac_bolus_dose_unit_consistent_{wmin}m'] = round(float(d[f'hist_same_bolus_dose_{wmin}m_unit_match'].mean()), 3)
        ha.append(row)
        for c in [c for c in d.columns if c.startswith('hist_') and d[c].dtype != object]:
            v = d[c].dropna()
            hs.append({'drug': drug, 'feature': c, 'n_nonmissing': int(len(v)),
                       'frac_missing': round(1 - len(v) / len(d), 3),
                       'mean': round(float(v.mean()), 3) if len(v) else np.nan,
                       'p50': round(float(v.median()), 3) if len(v) else np.nan,
                       'p90': round(float(v.quantile(.9)), 3) if len(v) else np.nan,
                       'frac_zero': round(float((v == 0).mean()), 3) if len(v) else np.nan})
    pd.DataFrame(ha).to_csv(OUT / "v12_history_availability.csv", index=False)
    pd.DataFrame(hs).to_csv(OUT / "v12_history_features_summary.csv", index=False)

    # ------------------------------------------------------------------
    # C. Follow-up census
    # ------------------------------------------------------------------
    fu = []
    for drug, d in ev.groupby('drug'):
        for wmin in FOLLOW_WINDOWS_MIN:
            for k in ['same_redose', 'other_vaso', 'new_infusion_other', 'rate_change_any']:
                col = f'fu_{k}_{wmin}m'
                r = {'drug': drug, 'window_min': wmin, 'intervention': k,
                     'frac_events': round(float(d[col].mean()), 3), 'n_events': int(d[col].sum())}
                tcol = f'{col}_first_min'
                if tcol in d.columns:
                    t = d.loc[d[col] == 1, tcol].dropna()
                    if len(t):
                        r.update({'time_median_min': round(float(t.median()), 2),
                                  'time_p25': round(float(t.quantile(.25)), 2),
                                  'time_p75': round(float(t.quantile(.75)), 2)})
                fu.append(r)
    pd.DataFrame(fu).to_csv(OUT / "v12_followup_census.csv", index=False)

    # ------------------------------------------------------------------
    # Infusion reconstruction feasibility
    # ------------------------------------------------------------------
    ir = []
    for drug, d in ev.groupby('drug'):
        inf = d[d.event_class_v12.isin(['infusion_start_or_bag_change', 'infusion_resume'])]
        ir.append({'drug': drug, 'n_infusion_type_events': len(inf),
                   'frac_with_prior_running_rate': round(float((inf.hist_running_rate_same > 0).mean()), 3) if len(inf) else np.nan,
                   'frac_rate_unit_recorded': round(float(inf.dose_unit.isin(RATE_UNITS).mean()), 3) if len(inf) else np.nan,
                   'order_id_column_present': int(bool(ocol)),
                   'median_running_rate_age_min': round(float(inf.hist_running_rate_age_min.median()), 1) if inf.hist_running_rate_age_min.notna().any() else np.nan})
    pd.DataFrame(ir).to_csv(OUT / "v12_infusion_reconstruction.csv", index=False)

    # ------------------------------------------------------------------
    # Window quality
    # ------------------------------------------------------------------
    wq = []
    for drug, d in ev.groupby('drug'):
        wq.append({'drug': drug, 'n': len(d),
                   'baseline_map_p01': round(float(d.baseline_map.quantile(.01)), 1),
                   'baseline_map_p99': round(float(d.baseline_map.quantile(.99)), 1),
                   'delta_abs_gt_60_frac': round(float((d[TARGET].abs() > 60).mean()), 4),
                   'frac_overlapping_outcome_windows': round(float((d['fu_same_redose_5m'] == 1).mean()), 3)})
    pd.DataFrame(wq).to_csv(OUT / "v12_window_quality.csv", index=False)

    # ------------------------------------------------------------------
    # D. Reconciliation with a previous extraction and patient-set overlap
    # ------------------------------------------------------------------
    if args.previous_events and Path(args.previous_events).exists():
        prev = pd.read_parquet(args.previous_events)
        key = [c for c in ['caseid', 'subjectid', 'drug', 'baseline_map', TARGET,
                            'ctx_surgery_time_min', 'ctx_map_mean', 'ctx_map_last']
               if c in prev.columns and c in ev.columns]
        logging.info(f"reconciliation key: {key}")
        # coerce join keys to a common dtype (tables from different runs may differ)
        for c in key:
            if c in ('caseid', 'subjectid', 'drug'):
                prev[c] = prev[c].astype(str); ev[c] = ev[c].astype(str)
        m = prev.merge(ev[key + ['event_id']], on=key, how='left', indicator=True)
        rec = m.groupby('drug')['_merge'].value_counts().unstack(fill_value=0).reset_index()
        rec['n_previous'] = prev.groupby('drug').size().reindex(rec.drug).values
        rec['n_current'] = ev.groupby('drug').size().reindex(rec.drug).values
        rec.to_csv(OUT / "v12_reconciliation.csv", index=False)
        logging.info(f"reconciliation with previous extraction:\n{rec}")
        pr = []
        for drug in ev.drug.unique():
            p11 = set(prev[prev.drug == drug].subjectid.astype(str).unique())
            p12 = set(ev[ev.drug == drug].subjectid.astype(str).unique())
            pr.append({'drug': drug, 'patients_previous': len(p11), 'patients_current': len(p12),
                       'patients_in_both': len(p11 & p12), 'new_in_current': len(p12 - p11),
                       'partition_reusable_without_new_patients': int(len(p12 - p11) == 0)})
        pd.DataFrame(pr).to_csv(OUT / "v12_partition_reuse.csv", index=False)
    else:
        logging.warning("no --previous_events supplied; reconciliation skipped")

    summary = {'generated': datetime.now().isoformat(), 'n_cases': len(cases),
               'n_events': int(len(ev)), 'n_patients': int(ev.subjectid.nunique()),
               'event_class_counts': {d: cls.loc[d].to_dict() for d in cls.index},
               'order_id_col_configured': bool(ocol), 'fluid_file_configured': bool(cfg.get('fluid_file')),
               'blood_file_configured': bool(cfg.get('blood_file'))}
    (OUT / "v12_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logging.info("=== extraction complete ===")


if __name__ == '__main__':
    main()
