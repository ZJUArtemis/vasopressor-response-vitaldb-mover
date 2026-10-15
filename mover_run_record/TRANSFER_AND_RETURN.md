# MOVER-server operator instructions — Topic 2 v3

You received `topic2_v3_mover_package.tar.gz`. Everything runs on CPU. The scripts
only ever write into `outputs/`. They never modify MOVER source files.

---

## Step 0 — unpack & install deps
```bash
tar xzf topic2_v3_mover_package.tar.gz
cd mover_package
python3 -m pip install numpy pandas pyarrow scikit-learn xgboost pyyaml   # if missing
```

## Step 1 — sanity check on synthetic data (optional but recommended)
```bash
bash run_all.sh --test
```
If this completes without a traceback, the pipeline wiring is fine. Then clean up:
```bash
rm -rf outputs/* fixtures/*.csv fixtures/mover_columns.yaml
```

## Step 2 — point the config at the real MOVER files
Edit `config/mover_columns.yaml`. You must set the four absolute paths and confirm the
column names match your MOVER export:

| key | what it is |
|-----|-----------|
| `surgery_table` | OR/surgery table with in/out timestamps |
| `obs_file`      | Patient Observations (long format: name, value, time) |
| `med_file`      | medication administration record (MAR) |
| `patient_file`  | demographics (age, sex, height, weight, ASA) |

Also confirm these column-name keys inside the same file:
`case_col`, `mrn_col`, `or_in_col`, `or_out_col`, `obs_name_col`, `obs_value_col`,
`obs_time_col`, `map_name_value` (the exact string for arterial MAP, e.g. `MAP-ART A-line`),
`bp_art_name_value`, `med_name_col`, `med_action_col`, `med_time_col`, `med_dose_col`,
`med_dose_unit_col`, and `vaso_name_patterns` (lowercase substrings per drug).

The timestamp format default is `%m/%d/%y %H:%M`; the parser also tries ISO8601.

## Step 3 — run the GO/PAUSE gate FIRST
```bash
bash run_all.sh --gate
cat outputs/m0_dose_audit.json
```
Read the verdict:
- **`VERDICT: GO`** — G1 (timestamp parse ≥90%), G2 (dose parse ≥80%), G3 (every MAR
  action string is classified). Proceed to Step 4.
- **`VERDICT: PAUSE`** — do NOT proceed. Most common cause is G3: unmapped MAR action
  strings listed under `G3_unmapped_actions`. Classify each one in
  `config/mover_mar_actions.yaml` (into `keep_as_administration`, `keep_as_rate_change`,
  or `exclude`) and re-run `--gate`. If G1/G2 fail on the real data, STOP and report
  back — per the roadmap we pause Topic 2 rather than weaken the exposure definition.

## Step 4 — full run
```bash
bash run_all.sh --full
```
This runs:
1. `m0_dose_audit.py` again (re-confirms the gate),
2. `m1_extract_events.py` → `outputs/mover_documented_events.parquet`
   (G4 = ≥1000 START events with arterial MAP linkage; check the log line
   `START by drug`),
3. `m2_model_and_external.py` → MOVER-native models + VitalDB→MOVER external
   validation with linear + isotonic recalibration.

For a quick smoke run on a subset first:
```bash
python3 src/m1_extract_events.py --max_cases 500
```

## Step 5 — return results
Zip and send back the **entire `outputs/` directory**:
```bash
tar czf topic2_v3_mover_outputs.tar.gz outputs/
```
Key files I need:
- `outputs/m0_dose_audit.json` — the gate verdict + action vocabulary
- `outputs/mover_documented_events.parquet` (+ `_sample.csv`) — event table
- `outputs/m2_mover_native.csv` — MOVER-native per-drug MAE/calibration/PI coverage
- `outputs/m2_external_validation.csv` — VitalDB→MOVER transportability + recalibration
- `outputs/logs/*.log` — full run logs

**Do not** send any raw MOVER rows beyond the de-identified aggregates already in those
files. The event table contains per-event physiology (MAP, dose, age) keyed by an
internal case id — confirm this is within your DUA before transfer; if not, send only
the `m0`/`m2` CSV summaries and the logs.

---

### What each gate protects
The whole point of v3 is that exposure is **documented**, never inferred from the MAP
rise it is meant to predict. m0 is the formal checkpoint that the MOVER MAR actually
supports that. If it can't, the honest result is to pause — that is a valid outcome,
not a failure.
