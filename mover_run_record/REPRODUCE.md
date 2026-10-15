# Reproduce this local execution

The scripts in this directory are snapshots for review. Put the `local_*.py` scripts in the original package's `src/`, and `local_run.sh` in its root. Keep the original shipped VitalDB models unchanged. The input package and model SHA-256 digests are in RETURN_MANIFEST.json.

Install the dependencies in requirements_versions.txt for Python 3.12 (execution used Python 3.12.14). Original source archives remain read-only. Archive paths in local_prepare.py and local_gate_audit.py reflect the mounted dataset on this computer; adjust only paths if reproducing on another machine.

`local_run.sh --prepare` scans both complete archives and writes local intermediate CSVs under local_work/. It maps actual fields and writes config/mover_columns_local.yaml. This can take tens of minutes because the flowsheet archive is approximately 152 GB uncompressed.

`local_run.sh --analyze` rechecks the gate, extracts events, enforces G4, fits/evaluates models, validates tables and writes the final archive. It expects m0_dose_audit_initial.json from the initial full-source audit for a completeness cross-check. That initial file is included in the return package. If starting entirely from scratch, run local_gate_audit.py first to generate the initial census and preserve its JSON as m0_dose_audit_initial.json.

`local_run.sh --finalize` rebuilds the Chinese report and return archive from completed results.

Synthetic smoke verification ran in <TMPDIR>/topic2_v3_smoke, separate from the real package. synthetic_smoke_runner.py records the test procedure. The synthetic fixture has only 41 eligible START events: G4 deliberately fails, then the runner overrides ONLY its synthetic gate to exercise m2 at --min_n 5. The real G4 was never bypassed. SYNTHETIC_ONLY_smoke.log is test evidence, not real-data inference.

The native and external patient-level prediction parquet files contain oof_fold. local_report.py checks that a patient's events do not span test folds within each per-drug analysis, and recomputes all reported MAEs from event-level predictions.

Do not replace the local scripts with the originals when comparing final results: the documented method changes affect native predictors and honest recalibration evaluation.
