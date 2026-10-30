#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared loading and windowing code for the MOVER host extraction.

Reads the MOVER surgery, medication-administration and arterial-observation tables
forward-only, keeps the raw arterial MAP series per case, and computes the pre-dose
and post-dose window statistics used to define baseline MAP and the 5-minute outcome.
Writes nothing. Never modifies any source file.
"""
import os, sys, json, logging, argparse, itertools
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "outputs"

FEATURES_COMMON = ['age', 'sex_male', 'bmi', 'asa', 'dose_rate_at_event',
                   'baseline_map', 'ctx_map_mean', 'ctx_map_min', 'ctx_map_last',
                   'ctx_map_trend', 'ctx_map_cv', 'ctx_time_below_65_min',
                   'ctx_surgery_time_min']
TARGET = 'delta_map_5min'
GROUP = 'subjectid'
STRICT_BASELINE_WINDOW = False   # False reproduces the published analysis
MAP_MIN, MAP_MAX = 20, 300              # arterial MAP physiologic range (artefact rule)


def setup_log(name):
    OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(OUT / "logs" / f"{ts}_{name}.log"), logging.StreamHandler()])


def load_cfg(cfg_path=None):
    cfg = yaml.safe_load(open(cfg_path or (PKG / "config" / "mover_columns.yaml")))
    actions = yaml.safe_load(open(PKG / "config" / "mover_mar_actions.yaml"))
    return cfg, actions


def parse_time(series):
    dt = pd.to_datetime(series, errors='coerce', format='ISO8601')
    missing = dt.isna() & series.notna()
    dt.loc[missing] = pd.to_datetime(series.loc[missing], errors='coerce', format='%m/%d/%y %H:%M')
    return dt


def match_drug(name, patterns):
    n = str(name).lower()
    for drug, pats in sorted(patterns.items(), key=lambda it: it[0] != 'norepinephrine'):
        if any(p in n for p in pats):
            return drug
    return None

def load_raw(cfg, actions, max_cases=0):
    """Return (med_by_case, map_by_case, win, pat, ccol). Raw MAP kept per case."""
    ccol = cfg['case_col']
    surg = pd.read_csv(cfg['surgery_table'])
    surg['_in'] = parse_time(surg[cfg['or_in_col']])
    surg['_out'] = parse_time(surg[cfg['or_out_col']])
    surg = surg.dropna(subset=['_in', '_out'])
    win = surg.set_index(ccol)[['_in', '_out']].to_dict('index')
    logging.info(f"surgery windows: {len(win)}")

    med = pd.read_csv(cfg['med_file'])
    med['_drug'] = med[cfg['med_name_col']].map(lambda x: match_drug(x, cfg['vaso_name_patterns']))
    med = med[med['_drug'].notna()].copy()
    keep_admin = set(actions['keep_as_administration'])
    keep_rate = set(actions['keep_as_rate_change'])
    act = med[cfg['med_action_col']].astype(str)
    med['_etype'] = np.where(act.isin(keep_admin), 'start',
                     np.where(act.isin(keep_rate), 'rate_increase', 'drop'))
    med['_action'] = act
    med = med[med['_etype'] != 'drop'].copy()
    med['_t'] = parse_time(med[cfg['med_time_col']])
    med = med.dropna(subset=['_t'])
    logging.info(f"documented vasopressor MAR rows (admin+rate): {len(med)}")

    eligible = set(win).intersection(set(med[ccol]))
    parts = []
    for part in pd.read_csv(cfg['obs_file'], chunksize=250000):
        part = part[part[cfg['obs_name_col']].eq(cfg['map_name_value']) & part[ccol].isin(eligible)]
        if len(part):
            parts.append(part)
    obs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    del parts
    obs['_t'] = parse_time(obs[cfg['obs_time_col']])
    obs['_map_raw'] = pd.to_numeric(obs[cfg['obs_value_col']], errors='coerce')
    logging.info(f"raw arterial MAP rows (pre-filter): {len(obs)}")

    pat = pd.read_csv(cfg['patient_file']).set_index(ccol) if cfg.get('patient_file') else None

    med_by = {c: d.sort_values('_t') for c, d in med.groupby(ccol)}
    map_by = {c: d.sort_values('_t') for c, d in obs.groupby(ccol)}
    cases = list(med_by.keys())
    if max_cases:
        cases = cases[:max_cases]
    return med_by, {c: map_by.get(c) for c in cases}, win, pat, ccol, cases


def window_stats(mapcase, t0, offset_min=0, pre_s=600, post_s=600):
    """Pre/post-dose window statistics around t0, with an optional forward baseline offset.

    offset_min>0 shifts the baseline anchor forward: baseline is the median MAP over
    the 2 min BEFORE (t0 + offset), while the outcome stays anchored at the true t0.
    offset_min=0 is the definition used in the study. MAP is range-filtered
    here (not upstream) so the same raw series serves the signal-quality census.
    """
    mc = mapcase[(mapcase['_map_raw'] >= MAP_MIN) & (mapcase['_map_raw'] <= MAP_MAX)].copy()
    mc = mc.rename(columns={'_map_raw': '_map'})
    t_base = t0 + pd.Timedelta(minutes=offset_min)
    pre = mc[(mc['_t'] >= t_base - pd.Timedelta(seconds=pre_s)) & (mc['_t'] < t_base)]
    post = mc[(mc['_t'] >= t0) & (mc['_t'] <= t0 + pd.Timedelta(seconds=post_s))]
    if len(pre) < 6:
        return None
    pre_last2 = pre[pre['_t'] >= t_base - pd.Timedelta(seconds=120)]
    # Published analysis: when the 2-minute window held no sample, the 10-minute
    # pre-window median was used instead of excluding the administration (0.69% of
    # analysed administrations; see Supplementary Table S16). Set STRICT_BASELINE_WINDOW
    # to True to exclude such administrations instead.
    if len(pre_last2):
        baseline = float(pre_last2['_map'].median())
    elif STRICT_BASELINE_WINDOW:
        return None
    else:
        baseline = float(pre['_map'].median())
    if not np.isfinite(baseline):
        return None
    seg5 = post[(post['_t'] >= t0 + pd.Timedelta(seconds=240)) & (post['_t'] <= t0 + pd.Timedelta(seconds=360))]
    if len(seg5) < 1:
        return None
    dmap5 = float(seg5['_map'].mean()) - baseline
    mins = (pre['_t'] - t_base).dt.total_seconds().values / 60.0
    vals = pre['_map'].values
    trend = np.polyfit(mins, vals, 1)[0] if len(vals) > 3 else np.nan
    return {
        'baseline_map': round(baseline, 2),
        'delta_map_5min': round(dmap5, 2),
        'ctx_map_mean': round(float(np.mean(vals)), 2),
        'ctx_map_min': round(float(np.min(vals)), 2),
        'ctx_map_last': round(float(vals[-1]), 2),
        'ctx_map_trend': round(float(trend), 3) if np.isfinite(trend) else None,
        'ctx_map_cv': round(float(np.std(vals) / np.mean(vals)), 4) if np.mean(vals) else None,
        'ctx_time_below_65_min': round(float(np.sum(vals < 65)) * (pre_s / 60.0) / max(len(vals), 1), 2),
    }
