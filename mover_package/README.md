# Topic 2 v3 — MOVER analysis package

**Study**: Prediction of observed arterial pressure response following *documented*
intraoperative vasopressor administration. (NOT a causal treatment-effect claim.)

**Why this package exists**: VitalDB is the development cohort but is too small
(≈120 phenylephrine + ≈66 norepinephrine documented START events, no ephedrine pump
track). MOVER (UC Irvine) carries the headline results because its medication
administration record (MAR) gives *actually-recorded* administration times, drug
names, and doses — a genuine documented exposure, not a MAP-inferred one.

This directory is self-contained and runs on the MOVER server with CPU only.
It never writes outside `outputs/`.

## Layout
```
mover_package/
├── run_all.sh              # orchestrator: --test | --gate | --full
├── TRANSFER_AND_RETURN.md  # step-by-step for the MOVER-server operator (READ THIS)
├── config/
│   ├── mover_columns.yaml      # EDIT: file paths + column names
│   └── mover_mar_actions.yaml  # EDIT: MAR action -> keep/exclude classification
├── src/
│   ├── m0_dose_audit.py        # GO/PAUSE gate (G1 timestamp, G2 dose, G3 action vocab)
│   ├── m1_extract_events.py    # documented event table (mirrors VitalDB v3_step1)
│   ├── m2_model_and_external.py# MOVER-native models + VitalDB->MOVER external validation
│   └── make_fixture.py         # synthetic clean-room test data (no real MOVER)
├── models/vitaldb/             # frozen VitalDB bundles for external validation
│   ├── phenylephrine.pkl       # n=120
│   └── norepinephrine.pkl      # n=66
├── fixtures/                   # (empty; populated by make_fixture.py during --test)
└── outputs/                    # (empty; all results land here — return this folder)
```

## Dependencies
`python3` with `numpy pandas pyarrow scikit-learn xgboost pyyaml`.

## One-line sanity check (no MOVER data needed)
```bash
bash run_all.sh --test
```
Should run m0→m1→m2 on synthetic data and finish without error.

Full instructions: see **TRANSFER_AND_RETURN.md**.
