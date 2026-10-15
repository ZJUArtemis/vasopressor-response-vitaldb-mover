#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_fixture.py | Topic 2 v3 | Build synthetic MOVER-shaped CSVs to clean-room test
the m0/m1/m2 pipeline WITHOUT touching real MOVER data. Writes to fixtures/ and a
matching fixtures/mover_columns.yaml pointing at them. Not for science — only to
prove the scripts run end-to-end and the schema wiring is correct.
"""
import os, numpy as np, pandas as pd, yaml
from pathlib import Path
from datetime import datetime, timedelta

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
FX = PKG / "fixtures"
FX.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

N_CASES = 60
drugs = {'phenylephrine': 'Phenylephrine 100 mcg/mL', 'ephedrine': 'Ephedrine 5 mg/mL',
         'norepinephrine': 'Norepinephrine (Levophed)'}
base = datetime(2019, 6, 1, 8, 0, 0)

surg, obs, med, pat = [], [], [], []
for c in range(1, N_CASES + 1):
    log_id = 1000 + c
    mrn = 50000 + (c % 45)  # some shared patients
    t_in = base + timedelta(days=c, minutes=int(rng.integers(0, 120)))
    dur = int(rng.integers(90, 240))
    t_out = t_in + timedelta(minutes=dur)
    surg.append({'LOG_ID': log_id, 'IN_OR_DTTM': t_in.strftime('%m/%d/%y %H:%M'),
                 'OUT_OR_DTTM': t_out.strftime('%m/%d/%y %H:%M')})
    pat.append({'LOG_ID': log_id, 'MRN': mrn, 'AGE': int(rng.integers(30, 85)),
                'SEX': rng.choice(['M', 'F']), 'HEIGHT': float(rng.integers(150, 190)),
                'WEIGHT': float(rng.integers(50, 100)), 'ASA_RATING_C': int(rng.integers(1, 4))})
    # arterial MAP every 60 s across surgery
    map0 = rng.normal(78, 8)
    for k in range(dur):
        tt = t_in + timedelta(minutes=k)
        mval = map0 + rng.normal(0, 4) - (6 if 40 < k < 70 else 0)  # a dip mid-case
        obs.append({'LOG_ID': log_id, 'RECORDED_TIME': tt.isoformat(timespec='seconds'),
                    'FLO_DISPLAY_NAME': 'MAP-ART A-line', 'MEAS_VALUE': round(float(mval), 1)})
        sbp = mval + rng.normal(35, 5); dbp = mval - rng.normal(18, 4)
        obs.append({'LOG_ID': log_id, 'RECORDED_TIME': tt.isoformat(timespec='seconds'),
                    'FLO_DISPLAY_NAME': 'BP-ART A-line', 'MEAS_VALUE': f"{int(sbp)}/{int(dbp)}"})
    # a couple of documented vasopressor pushes during the dip
    for k in [50, 60]:
        d = rng.choice(list(drugs))
        tt = t_in + timedelta(minutes=k)
        med.append({'LOG_ID': log_id, 'MEDICATION_NM': drugs[d],
                    'ADMIN_TIME': tt.isoformat(timespec='seconds'),
                    'DOSE': float(rng.integers(5, 100)), 'DOSE_UNIT': 'mcg',
                    'ROUTE': 'IV', 'MAR_ACTION': rng.choice(['Given', 'Bolus', 'Rate Change', 'Due'])})

pd.DataFrame(surg).to_csv(FX / "surgery.csv", index=False)
pd.DataFrame(obs).to_csv(FX / "observations.csv", index=False)
pd.DataFrame(med).to_csv(FX / "medications.csv", index=False)
pd.DataFrame(pat).to_csv(FX / "patients.csv", index=False)

cfg = yaml.safe_load(open(PKG / "config" / "mover_columns.yaml"))
cfg['surgery_table'] = str(FX / "surgery.csv")
cfg['obs_file'] = str(FX / "observations.csv")
cfg['med_file'] = str(FX / "medications.csv")
cfg['patient_file'] = str(FX / "patients.csv")
yaml.safe_dump(cfg, open(FX / "mover_columns.yaml", 'w'))
print(f"fixture written to {FX} ({len(obs)} obs rows, {len(med)} med rows, {N_CASES} cases)")
