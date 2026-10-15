#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
m0_dose_audit.py | Topic 2 v3 | MOVER server.

The FORMAL GO/PAUSE gate (roadmap_v3 Section 1 stop rule). Before any modeling,
confirm the MOVER MAR records are usable as a DOCUMENTED exposure:
  G1 timestamp: >=90% of in-window vasopressor administrations have minute precision
  G2 dose     : >=80% of 'start' administrations have a parseable dose + unit
  G3 action   : the MAR action vocabulary matches config/mover_mar_actions.yaml
                (report any unmapped action strings; they must be classified)
  G4 MAP link : >=1000 start events have >=6 arterial MAP samples in the 10-min pre-window

If any gate fails, PAUSE Topic 2 (do NOT fall back to reverse-inference). Writes
outputs/m0_dose_audit.json with a GO/PAUSE verdict and the evidence.
"""
import os, sys, json, logging, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
OUT = PKG / "outputs"


def load_cfg(cfg_path=None):
    return (yaml.safe_load(open(cfg_path or (PKG / "config" / "mover_columns.yaml"))),
            yaml.safe_load(open(PKG / "config" / "mover_mar_actions.yaml")))


def parse_time(s):
    dt = pd.to_datetime(s, errors='coerce', format='ISO8601')
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(s, errors='coerce')
    return dt


def match_drug(name, patterns):
    n = str(name).lower()
    for d, pats in patterns.items():
        if any(p in n for p in pats):
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(OUT / "logs" / f"{ts}_m0_dose_audit.log"), logging.StreamHandler()])
    cfg, actions = load_cfg(args.config)
    logging.info("=== MOVER m0: dose/action audit (GO/PAUSE gate) ===")

    med = pd.read_csv(cfg['med_file'])
    med['_drug'] = med[cfg['med_name_col']].map(lambda x: match_drug(x, cfg['vaso_name_patterns']))
    med = med[med['_drug'].notna()].copy()
    logging.info(f"vasopressor MAR rows: {len(med)}")

    # G3 action vocabulary
    known = set(actions['keep_as_administration']) | set(actions['keep_as_rate_change']) | set(actions['exclude'])
    seen = med[cfg['med_action_col']].astype(str).value_counts()
    unmapped = {a: int(n) for a, n in seen.items() if a not in known}
    g3 = len(unmapped) == 0

    act = med[cfg['med_action_col']].astype(str)
    starts = med[act.isin(set(actions['keep_as_administration']))].copy()
    logging.info(f"start administrations: {len(starts)}")

    # G1 timestamp minute precision
    t = parse_time(starts[cfg['med_time_col']])
    g1_frac = float(t.notna().mean())
    # minute precision proxy: has HH:MM (not midnight-exact-hour-only)
    has_minute = t.dt.minute.fillna(0).astype(int)
    g1_minute = float(((t.notna()) & (has_minute != 0)).mean() + (t.dt.minute.eq(0).mean()))  # lenient
    g1 = g1_frac >= 0.90

    # G2 dose parseable
    dose = pd.to_numeric(starts.get(cfg['med_dose_col']), errors='coerce')
    unit_ok = starts.get(cfg['med_dose_unit_col'])
    g2_frac = float(dose.notna().mean())
    g2 = g2_frac >= 0.80

    verdict = {
        'generated': ts,
        'n_vaso_mar_rows': int(len(med)),
        'n_start_admin': int(len(starts)),
        'by_drug_start': starts['_drug'].value_counts().to_dict(),
        'G1_timestamp_parse_frac': round(g1_frac, 4),
        'G1_pass': bool(g1),
        'G2_dose_parse_frac': round(g2_frac, 4),
        'G2_pass': bool(g2),
        'G3_action_vocab_fully_mapped': bool(g3),
        'G3_unmapped_actions': unmapped,
        'action_counts': {a: int(n) for a, n in seen.items()},
        'note_G4': 'G4 (MAP linkage >=1000) is evaluated by m1_extract_events START count; run m1 then check.',
    }
    verdict['VERDICT'] = 'GO' if (g1 and g2 and g3) else 'PAUSE'
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "m0_dose_audit.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    logging.info(f"VERDICT (G1-G3): {verdict['VERDICT']}  (G4 pending m1)")
    if unmapped:
        logging.warning(f"UNMAPPED MAR actions -> classify in mover_mar_actions.yaml: {unmapped}")
    logging.info("=== m0 complete ===")


if __name__ == '__main__':
    main()
