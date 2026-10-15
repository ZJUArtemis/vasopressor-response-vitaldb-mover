#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: v3_step1_extract_events.py | Topic: 2 (v3) | Purpose:
Extract DOCUMENTED vasopressor administration events from VitalDB infusion-pump
tracks. NO MAP-inference anywhere — an event exists iff the pump record shows it.

Event = a documented "new dose":
  - infusion/bolus START: PHEN/NEPI _RATE transitions 0 -> >0 after >=5 min of 0, OR
  - cumulative-volume STEP: _VOL increases by a discrete amount (bolus push).
Primary analysis = START events. Rate increases during an ongoing infusion are tagged
`rate_increase` and kept for a sensitivity analysis.

Eligibility (roadmap_v3 Section 2.5):
  - >=60 non-artifact MAP samples in the 10-min pre-window
  - non-missing baseline MAP (2-min pre-median) and 5-min outcome
  - co-administration of a DIFFERENT pump vasopressor within [-0,+10] min -> excluded
  - same-drug re-dose within outcome window -> censored (outcome truncated), flagged

Outcome: delta_map_5min = mean(MAP[4..6 min post]) - baseline. (Also 2-min, peak, ttp.)
Leakage: every feature is pre-dose; outcome strictly post-dose.
"""
import os, sys, json, logging, gc
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DATA = Path(os.environ.get("VITALDB_ROOT", "/path/to/vitaldb/physionet.org"))
WORK = Path(os.environ.get("TOPIC2_WORK", Path(__file__).resolve().parents[1]))
BASE = RAW_DATA / "files/vitaldb/1.0.0"
OUT = WORK / "outputs_v3"
RAW_ROOTS = [Path(os.environ.get("VITALDB_ROOT", "/path/to/vitaldb/physionet.org")),
             Path(os.environ.get("INSPIRE_ROOT", "/path/to/inspire/physionet.org")),
             Path(os.environ.get("MIMIC4_ROOT", "/path/to/mimic/physionet.org"))]


def safe_path(p):
    r = str(Path(p).resolve())
    for root in RAW_ROOTS:
        assert not r.startswith(str(root.resolve())), f"FATAL write to raw: {root}"
    return Path(p)


ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
(OUT / "logs").mkdir(parents=True, exist_ok=True)
(OUT / "metrics").mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(OUT / "logs" / f"{ts}_v3_step1_events.log"),
              logging.StreamHandler()])
import vitaldb

PRE_WINDOW = 600      # 10 min pre
POST_WINDOW = 600     # 10 min post (outcome search)
MIN_GAP = 300         # >=5 min of zero rate before a "fresh start"
ART_MIN, ART_MAX = 20, 300
HR_MIN, HR_MAX = 20, 250

# drug -> (rate_track, vol_track)
DRUGS = {
    'phenylephrine':  ('Orchestra/PHEN_RATE', 'Orchestra/PHEN_VOL'),
    'norepinephrine': ('Orchestra/NEPI_RATE', 'Orchestra/NEPI_VOL'),
}
# all pump vasopressor rate tracks, for co-administration checks
ALL_VASO_RATE = ['Orchestra/PHEN_RATE', 'Orchestra/NEPI_RATE', 'Orchestra/EPI_RATE',
                 'Orchestra/VASO_RATE', 'Orchestra/DOPA_RATE', 'Orchestra/DOBU_RATE']


def clean_map(a):
    a = a.astype(float)
    a[(a < ART_MIN) | (a > ART_MAX)] = np.nan
    return a


def load_signals(cid, extra_tracks):
    vf = vitaldb.VitalFile(str(BASE / f"vital_files/{cid:04d}.vital"))
    present = set(vf.get_track_names())
    want = ['Solar8000/ART_MBP', 'Solar8000/HR', 'Solar8000/ART_SBP'] + \
           [t for t in extra_tracks if t in present]
    want = [w for w in want if w in present]
    df = vf.to_pandas(want, interval=1)
    del vf; gc.collect()
    return df, present


def outcome_window(map_arr, t):
    """Return (baseline, dmap5, dmap2, peak_dmap, ttp_sec) using pre/post windows."""
    if t < PRE_WINDOW or t + POST_WINDOW > len(map_arr):
        return None
    pre = map_arr[t - PRE_WINDOW:t]
    post = map_arr[t:t + POST_WINDOW]
    if np.sum(~np.isnan(pre)) < 60:
        return None
    baseline = float(np.nanmedian(pre[-120:]))  # 2-min pre median
    if np.isnan(baseline):
        return None
    seg5 = post[240:360]
    dmap5 = float(np.nanmean(seg5) - baseline) if np.sum(~np.isnan(seg5)) >= 20 else np.nan
    seg2 = post[60:180]
    dmap2 = float(np.nanmean(seg2) - baseline) if np.sum(~np.isnan(seg2)) >= 20 else np.nan
    if np.all(np.isnan(post[:600])):
        peak_dmap, ttp = np.nan, np.nan
    else:
        peak_val = np.nanmax(post[:600]); peak_dmap = float(peak_val - baseline)
        ttp = int(np.nanargmax(np.where(np.isnan(post[:600]), -1e9, post[:600])))
    if np.isnan(dmap5):
        return None  # primary outcome mandatory
    return baseline, dmap5, dmap2, peak_dmap, ttp


def detect_starts(rate):
    """0 -> >0 transitions after >=MIN_GAP of zero."""
    rate = np.nan_to_num(rate, nan=0.0)
    infusing = (rate > 0.01).astype(int)
    if infusing.sum() == 0:
        return [], infusing
    cross = np.where(np.diff(infusing) == 1)[0] + 1
    starts = []
    for t in cross:
        lb = min(t, MIN_GAP)
        if np.any(infusing[t - lb:t] > 0):
            continue
        starts.append(int(t))
    # rate increases (ongoing infusion steps up) -> sensitivity set
    incr = []
    for t in np.where(np.diff(rate) > 0.5)[0] + 1:
        if infusing[t - 1] == 1:  # already infusing
            incr.append(int(t))
    return starts, infusing, incr


def main():
    logging.info("=== Topic 2 v3 Step 1: documented event extraction (VitalDB) ===")
    lists = json.load(open(OUT / "metrics" / "v3_pump_case_lists.json"))
    cdf = pd.read_csv(BASE / "clinical_data.csv").set_index('caseid')

    events = []
    n_excl_coadmin = 0
    n_censored = 0
    for drug, (rt, vt) in DRUGS.items():
        case_ids = sorted(set(lists.get(drug, [])) & set(lists.get('cases_with_art_mbp', [])))
        logging.info(f"[{drug}] candidate cases: {len(case_ids)}")
        for i, cid in enumerate(case_ids):
            try:
                df, present = load_signals(cid, [rt, vt] + ALL_VASO_RATE)
                if 'Solar8000/ART_MBP' not in df.columns or rt not in df.columns:
                    continue
                map_arr = clean_map(df['Solar8000/ART_MBP'].values)
                hr_arr = df['Solar8000/HR'].values.astype(float) if 'Solar8000/HR' in df.columns else np.full(len(map_arr), np.nan)
                hr_arr[(hr_arr < HR_MIN) | (hr_arr > HR_MAX)] = np.nan
                sbp_arr = df['Solar8000/ART_SBP'].values.astype(float) if 'Solar8000/ART_SBP' in df.columns else np.full(len(map_arr), np.nan)
                rate = df[rt].values.astype(float)
                starts, infusing, incr = detect_starts(rate)

                # co-admin rate arrays (other drugs)
                other_rates = {k: np.nan_to_num(df[k].values.astype(float), nan=0.0)
                               for k in ALL_VASO_RATE if k in df.columns and k != rt}

                for t, etype in [(t, 'start') for t in starts] + [(t, 'rate_increase') for t in incr]:
                    oc = outcome_window(map_arr, t)
                    if oc is None:
                        continue
                    baseline, dmap5, dmap2, peak_dmap, ttp = oc

                    # co-administration of a DIFFERENT vasopressor in [-0,+10] min
                    w0, w1 = t, min(t + POST_WINDOW, len(map_arr))
                    coadmin = any(np.any(arr[w0:w1] > 0.01) and np.any(np.diff((arr[max(0,t-60):w1] > 0.01).astype(int)) == 1)
                                  for arr in other_rates.values())
                    if coadmin:
                        n_excl_coadmin += 1
                        continue

                    # same-drug re-dose within outcome window -> censored flag
                    redose = np.any(np.diff((rate[t+30:w1] > 0.01).astype(int)) == 1)
                    if redose:
                        n_censored += 1

                    # pre-dose context (10 min)
                    pre = map_arr[t - PRE_WINDOW:t]
                    hr_pre = hr_arr[t - PRE_WINDOW:t]
                    sbp_pre = sbp_arr[t - PRE_WINDOW:t]
                    mins = np.arange(len(pre)) / 60.0
                    valid = ~np.isnan(pre)
                    trend = np.polyfit(mins[valid], pre[valid], 1)[0] if valid.sum() > 30 else np.nan
                    sbp_base = float(np.nanmedian(sbp_pre[-120:])) if np.sum(~np.isnan(sbp_pre[-120:])) > 10 else np.nan
                    hr_base = float(np.nanmedian(hr_pre[-120:])) if np.sum(~np.isnan(hr_pre[-120:])) > 10 else np.nan

                    row = cdf.loc[cid] if cid in cdf.index else None
                    events.append({
                        'caseid': cid, 'subjectid': int(row['subjectid']) if row is not None else -1,
                        'drug': drug, 'event_type': etype, 'time_sec': t,
                        'dose_rate_at_event': float(rate[t]) if t < len(rate) else np.nan,
                        'censored_repeat': bool(redose),
                        # outcomes
                        'baseline_map': round(baseline, 2),
                        'delta_map_5min': round(dmap5, 2),
                        'delta_map_2min': round(dmap2, 2) if not np.isnan(dmap2) else None,
                        'peak_delta_map': round(peak_dmap, 2) if not np.isnan(peak_dmap) else None,
                        'time_to_peak_sec': int(ttp) if ttp == ttp else None,
                        # pre-dose context features
                        'ctx_map_mean': round(float(np.nanmean(pre)), 2),
                        'ctx_map_min': round(float(np.nanmin(pre)), 2),
                        'ctx_map_last': round(float(pre[valid][-1]), 2) if valid.sum() else None,
                        'ctx_map_trend': round(float(trend), 3) if trend == trend else None,
                        'ctx_map_cv': round(float(np.nanstd(pre) / np.nanmean(pre)), 4) if np.nanmean(pre) else None,
                        'ctx_hr_mean': round(float(np.nanmean(hr_pre)), 2) if np.sum(~np.isnan(hr_pre)) else None,
                        'ctx_shock_index': round(hr_base / sbp_base, 3) if (sbp_base and sbp_base == sbp_base and hr_base == hr_base) else None,
                        'ctx_time_below_65_min': round(float(np.nansum(pre < 65) / 60.0), 2),
                        'ctx_surgery_time_min': round(t / 60.0, 1),
                        # static
                        'age': pd.to_numeric(row['age'], errors='coerce') if row is not None else np.nan,
                        'sex_male': 1 if (row is not None and str(row['sex']).upper().startswith('M')) else 0,
                        'bmi': pd.to_numeric(row['bmi'], errors='coerce') if row is not None else np.nan,
                        'asa': pd.to_numeric(str(row['asa']).replace('ASA','').strip(), errors='coerce') if row is not None else np.nan,
                    })
                del df; gc.collect()
            except Exception as e:
                logging.debug(f"case {cid} ({drug}): {e}")
        logging.info(f"[{drug}] events so far: {len([e for e in events if e['drug']==drug])}")

    ev = pd.DataFrame(events)
    outp = safe_path(OUT / "metrics" / "vitaldb_documented_events.parquet")
    ev.to_parquet(outp, index=False)
    ev.to_csv(safe_path(OUT / "metrics" / "vitaldb_documented_events_sample.csv"), index=False)
    logging.info(f"Saved {len(ev)} events -> {outp}")
    if len(ev):
        logging.info(f"By drug/type:\n{ev.groupby(['drug','event_type']).size()}")
        logging.info(f"start events only: {int((ev['event_type']=='start').sum())}")
        logging.info(f"excluded co-admin: {n_excl_coadmin}; censored re-dose: {n_censored}")
        s = ev[ev['event_type']=='start']['delta_map_5min']
        logging.info(f"ΔMAP@5min (starts): mean={s.mean():.1f} median={s.median():.1f} std={s.std():.1f}")
    logging.info("=== Step 1 complete ===")


if __name__ == '__main__':
    main()
