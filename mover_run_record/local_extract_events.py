#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
m1_extract_events.py | Topic 2 v3 | MOVER server.

Build the DOCUMENTED vasopressor administration event table from the MOVER MAR,
mirroring the VitalDB v3_step1 logic so the two cohorts are directly comparable.

Exposure  : MAR rows whose action is a real administration (config/mover_mar_actions.yaml),
            inside the surgery window, minute-precision timestamp, per vasopressor.
Baseline  : median arterial MAP over the 2 min before ADMIN_TIME.
Outcome   : delta_map_5min = mean(MAP in [+4,+6] min) - baseline.
Context   : 10-min pre-dose MAP/HR features (same names as VitalDB).
Eligibility (mirror of VitalDB Step 1):
  - >=6 arterial MAP samples in the 10-min pre-window (MOVER MAP ~60 s resolution)
  - non-missing baseline + 5-min outcome
  - DIFFERENT-drug vasopressor within [-0,+10] min  -> exclude
  - same-drug re-dose within outcome window         -> redose_flag=1 (kept, covariate)
Output: outputs/mover_documented_events.parquet  (same schema as VitalDB events)

Run AFTER m0_dose_audit.py passes its GO gate. Pure CPU, pandas only.
"""
import os, sys, json, logging, argparse, re
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "outputs"


def load_cfg(cfg_path=None):
    cfg = yaml.safe_load(open(cfg_path or (PKG / "config" / "mover_columns.yaml")))
    actions = yaml.safe_load(open(PKG / "config" / "mover_mar_actions.yaml"))
    return cfg, actions


def setup_log(name):
    OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(OUT / "logs" / f"{ts}_{name}.log"), logging.StreamHandler()])


def parse_time(series):
    """Robust datetime parse: try ISO then %m/%d/%y %H:%M."""
    dt = pd.to_datetime(series, errors='coerce', format='ISO8601')
    missing = dt.isna() & series.notna()
    dt.loc[missing] = pd.to_datetime(series.loc[missing], errors='coerce', format='%m/%d/%y %H:%M')
    return dt


def match_drug(name, patterns):
    n = str(name).lower()
    for drug, pats in sorted(patterns.items(), key=lambda item: item[0] != 'norepinephrine'):
        if any(p in n for p in pats):
            return drug
    return None


def build_map_series(obs, cfg):
    """Return per-case arterial MAP as a dict[case] -> DataFrame(time, map, sbp, dbp)."""
    tname = cfg['obs_name_col']; tval = cfg['obs_value_col']
    tcol = cfg['obs_time_col']; ccol = cfg['case_col']
    obs = obs.copy()
    obs['_t'] = parse_time(obs[tcol])
    # arterial MAP rows
    is_map = obs[tname].astype(str) == cfg['map_name_value']
    mapdf = obs[is_map].copy()
    mapdf['_map'] = pd.to_numeric(mapdf[tval], errors='coerce')
    mapdf = mapdf[(mapdf['_map'] >= 20) & (mapdf['_map'] <= 300)]
    # BP-ART A-line "SBP/DBP" for shock index
    bp = obs[obs[tname].astype(str) == cfg.get('bp_art_name_value', '__none__')].copy()
    if len(bp):
        sp = bp[tval].astype(str).str.extract(r'(\d+)\s*/\s*(\d+)')
        bp['_sbp'] = pd.to_numeric(sp[0], errors='coerce')
        bp['_dbp'] = pd.to_numeric(sp[1], errors='coerce')
    return mapdf, bp, ccol


def window_stats(mapcase, t0, pre_s=600, post_s=600):
    """mapcase: DataFrame with _t,_map sorted. t0: Timestamp. Returns dict or None."""
    pre = mapcase[(mapcase['_t'] >= t0 - pd.Timedelta(seconds=pre_s)) & (mapcase['_t'] < t0)]
    post = mapcase[(mapcase['_t'] >= t0) & (mapcase['_t'] <= t0 + pd.Timedelta(seconds=post_s))]
    if len(pre) < 6:
        return None
    pre_last2 = pre[pre['_t'] >= t0 - pd.Timedelta(seconds=120)]
    baseline = float(pre_last2['_map'].median()) if len(pre_last2) else float(pre['_map'].median())
    if not np.isfinite(baseline):
        return None
    seg5 = post[(post['_t'] >= t0 + pd.Timedelta(seconds=240)) & (post['_t'] <= t0 + pd.Timedelta(seconds=360))]
    if len(seg5) < 1:
        return None
    dmap5 = float(seg5['_map'].mean()) - baseline
    # context
    mins = (pre['_t'] - t0).dt.total_seconds().values / 60.0
    vals = pre['_map'].values
    trend = np.polyfit(mins, vals, 1)[0] if len(vals) > 3 else np.nan
    peak = float(post['_map'].max()) - baseline if len(post) else np.nan
    return {
        'baseline_map': round(baseline, 2),
        'delta_map_5min': round(dmap5, 2),
        'peak_delta_map': round(peak, 2) if np.isfinite(peak) else None,
        'ctx_map_mean': round(float(np.mean(vals)), 2),
        'ctx_map_min': round(float(np.min(vals)), 2),
        'ctx_map_last': round(float(vals[-1]), 2),
        'ctx_map_trend': round(float(trend), 3) if np.isfinite(trend) else None,
        'ctx_map_cv': round(float(np.std(vals) / np.mean(vals)), 4) if np.mean(vals) else None,
        'ctx_time_below_65_min': round(float(np.sum(vals < 65)) * (pre_s / 60.0) / max(len(vals), 1), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_cases', type=int, default=0, help='0=all; else cap for a smoke test')
    ap.add_argument('--config', default=None, help='override config path (e.g. fixtures/mover_columns.yaml)')
    args = ap.parse_args()
    setup_log("m1_extract_events")
    gate = json.loads((OUT / "m0_dose_audit.json").read_text())
    if gate["VERDICT"] != "GO":
        raise SystemExit("PAUSE: G1-G3 gate has not passed")
    cfg, actions = load_cfg(args.config)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.info("=== MOVER m1: documented event extraction ===")

    # --- surgery windows ---
    surg = pd.read_csv(cfg['surgery_table'])
    surg['_in'] = parse_time(surg[cfg['or_in_col']])
    surg['_out'] = parse_time(surg[cfg['or_out_col']])
    surg = surg.dropna(subset=['_in', '_out'])
    win = surg.set_index(cfg['case_col'])[['_in', '_out']].to_dict('index')
    logging.info(f"surgery windows: {len(win)}")

    # --- medications (documented exposure) ---
    med = pd.read_csv(cfg['med_file'])
    med['_drug'] = med[cfg['med_name_col']].map(lambda x: match_drug(x, cfg['vaso_name_patterns']))
    med = med[med['_drug'].notna()].copy()
    keep_admin = set(actions['keep_as_administration'])
    keep_rate = set(actions['keep_as_rate_change'])
    act = med[cfg['med_action_col']].astype(str)
    med['_etype'] = np.where(act.isin(keep_admin), 'start',
                     np.where(act.isin(keep_rate), 'rate_increase', 'drop'))
    med = med[med['_etype'] != 'drop'].copy()
    med['_t'] = parse_time(med[cfg['med_time_col']])
    med = med.dropna(subset=['_t'])
    logging.info(f"documented vasopressor MAR rows (admin+rate): {len(med)}")

    # --- MAP observations ---
    obs_parts = []
    eligible_cases = set(win).intersection(set(med[cfg['case_col']]))
    for part in pd.read_csv(cfg['obs_file'], chunksize=250000):
        part = part[part[cfg['obs_name_col']].eq(cfg['map_name_value']) & part[cfg['case_col']].isin(eligible_cases)]
        if len(part): obs_parts.append(part)
    obs = pd.concat(obs_parts, ignore_index=True)
    del obs_parts
    mapdf, bp, ccol = build_map_series(obs, cfg)
    del obs
    logging.info(f"arterial MAP rows: {len(mapdf)} across {mapdf[ccol].nunique()} cases")

    # --- patient covariates ---
    pat = pd.read_csv(cfg['patient_file']).set_index(cfg['case_col']) if cfg.get('patient_file') else None

    # group med + map by case
    med_by = {c: d.sort_values('_t') for c, d in med.groupby(cfg['case_col'])}
    map_by = {c: d.sort_values('_t') for c, d in mapdf.groupby(ccol)}
    cases = list(med_by.keys())
    if args.max_cases:
        cases = cases[:args.max_cases]

    rows = []
    n_excl, n_redose = 0, 0
    for ci, case in enumerate(cases):
        if (ci + 1) % 2000 == 0:
            logging.info(f"  {ci+1}/{len(cases)} cases | events={len(rows)}")
        if case not in map_by or case not in win:
            continue
        mapcase = map_by[case]
        w = win[case]
        mc = med_by[case]
        for _, ev in mc.iterrows():
            t0 = ev['_t']
            if not (w['_in'] <= t0 <= w['_out']):
                continue  # intraop only
            st = window_stats(mapcase, t0)
            if st is None:
                continue
            # different-drug co-admin within [-0,+10] min -> exclude
            others = mc[(mc['_drug'] != ev['_drug']) & (mc['_t'] >= t0) &
                        (mc['_t'] <= t0 + pd.Timedelta(seconds=600))]
            if len(others):
                n_excl += 1
                continue
            # same-drug re-dose within outcome window -> flag
            same = mc[(mc['_drug'] == ev['_drug']) & (mc['_t'] > t0 + pd.Timedelta(seconds=30)) &
                      (mc['_t'] <= t0 + pd.Timedelta(seconds=600))]
            redose = len(same) > 0
            if redose:
                n_redose += 1
            prow = pat.loc[case] if (pat is not None and case in pat.index) else None
            def pv(col):
                return pd.to_numeric(prow[cfg[col]], errors='coerce') if (prow is not None and cfg.get(col) in (prow.index if prow is not None else [])) else np.nan
            rec = {
                'caseid': case, 'subjectid': (prow[cfg['mrn_col']] if (prow is not None and cfg.get('mrn_col') in prow.index) else case),
                'drug': ev['_drug'], 'event_type': ev['_etype'],
                'dose_rate_at_event': pd.to_numeric(ev.get(cfg['med_dose_col']), errors='coerce'),
                'dose_unit': ev.get(cfg['med_dose_unit_col']),
                'med_route': ev.get(cfg['med_route_col']),
                'censored_repeat': bool(redose),
                'age': pv('age_col'), 'bmi': np.nan,
                'sex_male': 1 if (prow is not None and cfg.get('sex_col') in prow.index and str(prow[cfg['sex_col']]).upper().startswith('M')) else 0,
                'asa': pv('asa_col'),
                'ctx_hr_mean': None, 'ctx_shock_index': None,
                'ctx_surgery_time_min': round((t0 - w['_in']).total_seconds() / 60.0, 1),
            }
            # bmi from height/weight if present
            if prow is not None and cfg.get('height_col') in getattr(prow, 'index', []) and cfg.get('weight_col') in prow.index:
                h = pd.to_numeric(prow[cfg['height_col']], errors='coerce')
                wt = pd.to_numeric(prow[cfg['weight_col']], errors='coerce')
                if np.isfinite(h) and np.isfinite(wt) and h > 0:
                    rec['bmi'] = round(float(wt / ((h / 100.0) ** 2)), 1)
            rec.update(st)
            rows.append(rec)

    ev = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / "mover_documented_events.parquet"
    ev.to_parquet(outp, index=False)
    ev.head(200).to_csv(OUT / "mover_documented_events_sample.csv", index=False)
    logging.info(f"saved {len(ev)} events -> {outp}")
    if len(ev):
        logging.info(f"by drug/type:\n{ev.groupby(['drug','event_type']).size()}")
        st = ev[ev.event_type == 'start']
        logging.info(f"START by drug:\n{st.groupby('drug').size()}")
        logging.info(f"excluded diff-drug co-admin: {n_excl}; same-drug redose flagged: {n_redose}")
        logging.info(f"ΔMAP@5min (starts): mean={st['delta_map_5min'].mean():.1f} median={st['delta_map_5min'].median():.1f}")
    starts_n = int((ev['event_type'] == 'start').sum()) if len(ev) else 0
    gate['G4_START_events'] = starts_n
    gate['G4_pass'] = starts_n >= 1000
    gate['G4_status'] = 'GO' if starts_n >= 1000 else 'PAUSE'
    gate['OVERALL_VERDICT_G1_G4'] = gate['G4_status']
    (OUT / 'm0_dose_audit.json').write_text(json.dumps(gate, indent=2))
    logging.info("=== m1 complete; G4 %s ===", gate['G4_status'])
    if starts_n < 1000: raise SystemExit('PAUSE G4: fewer than 1000 MAP-linked START events')


if __name__ == '__main__':
    main()
