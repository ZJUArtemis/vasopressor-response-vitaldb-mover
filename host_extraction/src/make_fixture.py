#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_fixture.py | SYNTHETIC smoke-test data only.

Generates small FAKE MOVER-shaped CSVs + a matching config so the operator can verify
the pipeline wiring WITHOUT touching real MOVER data. All values are random; results
from this fixture are meaningless and must never appear in the manuscript.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
FIX = PKG / "fixtures"
FIX.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

N_CASE = 120
drugs = ['phenylephrine', 'ephedrine', 'norepinephrine', 'vasopressin', 'epinephrine']
base = pd.Timestamp('2020-01-01 08:00')

pat_rows, surg_rows, med_rows, obs_rows = [], [], [], []
for c in range(N_CASE):
    cid = 1000 + c
    mrn = 50000 + c
    t_in = base + pd.Timedelta(minutes=int(rng.integers(0, 600)))
    t_out = t_in + pd.Timedelta(minutes=int(rng.integers(90, 300)))
    pat_rows.append({'LOG_ID': cid, 'MRN': mrn, 'BIRTH_DATE': 1960 + int(rng.integers(0, 40)),
                     'SEX': rng.choice(['M', 'F']), 'HEIGHT_CM': 150 + rng.integers(0, 40),
                     'WEIGHT_KG': 50 + rng.integers(0, 50), 'ASA_RATING_C': int(rng.integers(1, 4))})
    surg_rows.append({'LOG_ID': cid, 'IN_OR_DTTM': t_in.strftime('%m/%d/%y %H:%M'),
                      'OUT_OR_DTTM': t_out.strftime('%m/%d/%y %H:%M')})
    # arterial MAP every ~60 s across the OR window
    t = t_in
    while t < t_out:
        obs_rows.append({'LOG_ID': cid, 'FLO_DISPLAY_NAME': 'MAP-ART A-line',
                         'MEAS_VALUE': float(np.clip(rng.normal(75, 12), 30, 160)),
                         'RECORDED_TIME': t.strftime('%m/%d/%y %H:%M')})
        t += pd.Timedelta(seconds=60)
    # 1-4 vasopressor administrations inside the window
    for _ in range(int(rng.integers(1, 5))):
        tt = t_in + pd.Timedelta(minutes=int(rng.integers(15, max(16, (t_out - t_in).seconds // 60 - 15))))
        med_rows.append({'LOG_ID': cid, 'MEDICATION_NM': rng.choice(drugs),
                         'MAR_ACTION_NM': rng.choice(['Given', 'Bolus', 'New Bag', 'Rate Change', 'Held', 'Due']),
                         'MED_ACTION_TIME': tt.strftime('%m/%d/%y %H:%M'),
                         'MED_ENTRY_TIME': (tt + pd.Timedelta(minutes=int(rng.integers(0, 12)))).strftime('%m/%d/%y %H:%M'),
                         'MED_MOD_TIME': (tt + pd.Timedelta(minutes=int(rng.integers(0, 30)))).strftime('%m/%d/%y %H:%M'),
                         'ADMIN_SIG': float(rng.integers(1, 200)), 'DOSE_UNIT_NM': 'mcg',
                         'MED_ROUTE_NM': 'IntraVENOUS'})

pd.DataFrame(pat_rows).to_csv(FIX / 'patient_information.csv', index=False)
pd.DataFrame(surg_rows).to_csv(FIX / 'patient_information_surg.csv', index=False)
pd.DataFrame(med_rows).to_csv(FIX / 'vasopressor_medications.csv', index=False)
pd.DataFrame(obs_rows).to_csv(FIX / 'arterial_observations.csv', index=False)

cfg = {
    'case_col': 'LOG_ID', 'mrn_col': 'MRN',
    'surgery_table': str(FIX / 'patient_information_surg.csv'),
    'or_in_col': 'IN_OR_DTTM', 'or_out_col': 'OUT_OR_DTTM',
    'obs_file': str(FIX / 'arterial_observations.csv'),
    'obs_time_col': 'RECORDED_TIME', 'obs_name_col': 'FLO_DISPLAY_NAME',
    'obs_value_col': 'MEAS_VALUE', 'map_name_value': 'MAP-ART A-line',
    'bp_art_name_value': 'BP-ART A-line', 'nibp_map_name_value': 'MAP-NIBP',
    'med_file': str(FIX / 'vasopressor_medications.csv'),
    'med_name_col': 'MEDICATION_NM', 'med_time_col': 'MED_ACTION_TIME',
    'med_dose_col': 'ADMIN_SIG', 'med_dose_unit_col': 'DOSE_UNIT_NM',
    'med_route_col': 'MED_ROUTE_NM', 'med_action_col': 'MAR_ACTION_NM',
    'vaso_name_patterns': {d: [d] for d in drugs},
    'patient_file': str(FIX / 'patient_information.csv'),
    'age_col': 'BIRTH_DATE', 'sex_col': 'SEX', 'height_col': 'HEIGHT_CM',
    'weight_col': 'WEIGHT_KG', 'asa_col': 'ASA_RATING_C',
    'med_entry_time_col': 'MED_ENTRY_TIME', 'med_mod_time_col': 'MED_MOD_TIME',
    'med_order_id_col': '', 'med_order_start_col': '', 'med_order_end_col': '',
    'fluid_file': '', 'fluid_time_col': '', 'fluid_volume_col': '',
    'blood_file': '', 'blood_time_col': '',
}
yaml.safe_dump(cfg, open(FIX / 'mover_columns.yaml', 'w'))
print('wrote synthetic fixture to', FIX)
