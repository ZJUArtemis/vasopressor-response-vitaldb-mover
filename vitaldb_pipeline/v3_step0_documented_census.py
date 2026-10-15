#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: v3_step0_documented_census.py | Topic: 2 (v3 redesign) | Purpose:
Establish the HONEST documented-vasopressor-administration cohort in VitalDB.

v3 redesign rationale
---------------------
The v1/v2 Topic 2 trained on 11,300 "events", of which 11,108 (98.3%) were
reverse-inferred from a MAP rise (topic7/src/step2_event_extraction.py
extract_indirect_eph_events). That is a circular definition: the outcome (ΔMAP)
was used to define the exposure (drug given), then the exposure was used to
predict the outcome. This script discards that approach entirely and counts ONLY
administrations that carry a genuine device timestamp:

  - Syringe/infusion-pump tracks (Orchestra/<DRUG>_RATE and _VOL). A pump push or
    infusion start is a real, recorded device event — no MAP inference involved.

Ephedrine in VitalDB is a manual IV bolus with NO pump track, hence NO timestamp.
It is therefore EXCLUDED from the VitalDB documented cohort (this is exactly why
v1/v2 had to infer it). Ephedrine returns only in MOVER, where it is documented in
the MAR with an administration time.

This script writes a census only (no modeling, no outcome use for exposure).
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
RAW_ROOTS = [
    Path(os.environ.get("VITALDB_ROOT", "/path/to/vitaldb/physionet.org")),
    Path(os.environ.get("INSPIRE_ROOT", "/path/to/inspire/physionet.org")),
    Path(os.environ.get("MIMIC4_ROOT", "/path/to/mimic/physionet.org")),
]


def safe_path(p):
    resolved = str(Path(p).resolve())
    for root in RAW_ROOTS:
        assert not resolved.startswith(str(root.resolve())), \
            f"FATAL: attempted write to raw data path: {root}"
    return Path(p)


ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
(OUT / "logs").mkdir(parents=True, exist_ok=True)
(OUT / "metrics").mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(OUT / "logs" / f"{ts}_v3_step0_census.log"),
              logging.StreamHandler()])

import vitaldb

# Pump-documented vasopressors in VitalDB (drug -> (rate_track, vol_track, clinical_col))
PUMP_DRUGS = {
    'phenylephrine':  ('Orchestra/PHEN_RATE', 'Orchestra/PHEN_VOL', 'intraop_phe'),
    'norepinephrine': ('Orchestra/NEPI_RATE', 'Orchestra/NEPI_VOL', None),
    'epinephrine':    ('Orchestra/EPI_RATE',  'Orchestra/EPI_VOL',  'intraop_epi'),
    'vasopressin':    ('Orchestra/VASO_RATE', 'Orchestra/VASO_VOL', None),
    'dopamine':       ('Orchestra/DOPA_RATE', 'Orchestra/DOPA_VOL', None),
    'dobutamine':     ('Orchestra/DOBU_RATE', 'Orchestra/DOBU_VOL', None),
}


def probe_tracks_fast(cid):
    """Header-only track list (vitaldb.vital_trks). ~6s/file.

    Used only for drugs NOT already covered by the Topic 7 census.
    """
    return set(vitaldb.vital_trks(str(BASE / f"vital_files/{cid:04d}.vital")))


def main():
    logging.info("=== Topic 2 v3 Step 0: documented-administration census ===")
    cdf = pd.read_csv(BASE / "clinical_data.csv")
    logging.info(f"Clinical cases: {len(cdf)}")

    drug_present_cases = {d: [] for d in PUMP_DRUGS}
    cases_with_art = []
    n_err = 0

    # --- FAST PATH: reuse Topic 7's already-computed pump-track census ---
    # topic7/outputs/metrics/drug_track_census.json holds phen/nepi pump-track
    # case lists and the ART_MBP (map_track) case list. Reusing it avoids a
    # ~10-hour re-probe of all 6,388 .vital headers. Falls back to probing only
    # for drugs the Topic 7 census does not cover (epi/vaso/dopa/dobu), and only
    # among cases the clinical table flags as having received that drug.
    t7 = WORK.parent / "topic7/outputs/metrics/drug_track_census.json"
    use_t7 = t7.exists() and os.environ.get("FORCE_PROBE", "0") != "1"
    if use_t7:
        c7 = json.load(open(t7))
        cases_with_art = list(c7.get('map_track_cases', []))
        drug_present_cases['phenylephrine'] = list(c7.get('phen_track_cases', []))
        drug_present_cases['norepinephrine'] = list(c7.get('nepi_track_cases', []))
        logging.info(f"Reused Topic 7 census: phen={len(drug_present_cases['phenylephrine'])} "
                     f"nepi={len(drug_present_cases['norepinephrine'])} "
                     f"art={len(cases_with_art)}")
        # Probe remaining pump drugs only among clinical candidates.
        remaining = ['epinephrine', 'vasopressin', 'dopamine', 'dobutamine']
        for d in remaining:
            _rt, _vt, ccol = PUMP_DRUGS[d]
            if ccol is not None and ccol in cdf.columns:
                cand = cdf[pd.to_numeric(cdf[ccol], errors='coerce').fillna(0) > 0]['caseid'].tolist()
            else:
                cand = []  # no clinical flag -> skip exhaustive probe (set FORCE_PROBE=1 for full)
            for cid in cand:
                try:
                    tracks = probe_tracks_fast(cid)
                    if _rt in tracks or _vt in tracks:
                        drug_present_cases[d].append(cid)
                except Exception as e:
                    n_err += 1
                    logging.debug(f"case {cid}: {e}")
    else:
        # --- SLOW PATH: full header probe (set FORCE_PROBE=1) ---
        caseids = cdf['caseid'].tolist()
        logging.info(f"FORCE_PROBE: probing {len(caseids)} case headers (slow)...")
        for i, cid in enumerate(caseids):
            if (i + 1) % 300 == 0:
                logging.info(f"  probed {i+1}/{len(caseids)} | ART={len(cases_with_art)}")
            try:
                tracks = probe_tracks_fast(cid)
                if 'Solar8000/ART_MBP' in tracks:
                    cases_with_art.append(cid)
                for d, (rt, vt, _c) in PUMP_DRUGS.items():
                    if rt in tracks or vt in tracks:
                        drug_present_cases[d].append(cid)
            except Exception as e:
                n_err += 1
                logging.debug(f"case {cid}: {e}")
            finally:
                if (i + 1) % 500 == 0:
                    gc.collect()

    art_set = set(cases_with_art)
    census = {
        'n_cases_probed': len(probe_ids),
        'n_errors': n_err,
        'n_cases_with_art_mbp': len(cases_with_art),
        'drugs': {},
    }
    for d in PUMP_DRUGS:
        present = drug_present_cases[d]
        present_with_art = [c for c in present if c in art_set]
        census['drugs'][d] = {
            'n_cases_with_pump_track': len(present),
            'n_cases_with_pump_track_and_art': len(present_with_art),
        }
        logging.info(f"{d:16s}: pump-track cases={len(present):5d} | "
                     f"with ART_MBP={len(present_with_art):5d}")

    out_json = safe_path(OUT / "metrics" / "v3_documented_census.json")
    with open(out_json, "w") as f:
        json.dump(census, f, indent=2)
    logging.info(f"Saved census -> {out_json}")

    # Save per-drug case lists for the next step (event extraction).
    lists = {d: drug_present_cases[d] for d in PUMP_DRUGS}
    lists['cases_with_art_mbp'] = cases_with_art
    out_lists = safe_path(OUT / "metrics" / "v3_pump_case_lists.json")
    with open(out_lists, "w") as f:
        json.dump(lists, f)
    logging.info(f"Saved case lists -> {out_lists}")
    logging.info("=== Step 0 complete ===")


if __name__ == '__main__':
    main()
