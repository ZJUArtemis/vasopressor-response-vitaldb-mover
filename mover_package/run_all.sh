#!/bin/bash
# Topic 2 v3 — MOVER server runner.
# Usage:
#   bash run_all.sh --test        # clean-room fixture test (no real MOVER data)
#   bash run_all.sh --gate        # m0 dose/action audit only (GO/PAUSE gate)
#   bash run_all.sh --full        # m0 -> m1 -> m2 on real MOVER (after editing config/)
#
# Prerequisites on the MOVER server:
#   - Python 3 with: numpy pandas pyarrow scikit-learn xgboost pyyaml
#   - Edit config/mover_columns.yaml: set the 4 ABS paths + verify column names.
#   - Verify config/mover_mar_actions.yaml covers all MAR action strings
#     (m0 reports any unmapped ones; add them before --full).
set -e
PY="${PY:-python}"
cd "$(dirname "$0")"

case "$1" in
  --test)
    echo "[test] building synthetic fixture + running m0/m1/m2 on it"
    $PY src/make_fixture.py
    $PY src/m0_dose_audit.py --config fixtures/mover_columns.yaml
    $PY src/m1_extract_events.py --config fixtures/mover_columns.yaml
    $PY src/m2_model_and_external.py --min_n 10
    echo "[test] OK if all three completed. Clean up: rm -rf outputs/* fixtures/*.csv fixtures/mover_columns.yaml"
    ;;
  --gate)
    $PY src/m0_dose_audit.py
    echo "[gate] see outputs/m0_dose_audit.json — proceed to --full only if VERDICT=GO"
    ;;
  --full)
    $PY src/m0_dose_audit.py
    $PY src/m1_extract_events.py            # add --max_cases 500 for a quick smoke run
    $PY src/m2_model_and_external.py
    echo "[full] done. Return the whole outputs/ directory."
    ;;
  *)
    echo "usage: bash run_all.sh [--test|--gate|--full]"; exit 1;;
esac
