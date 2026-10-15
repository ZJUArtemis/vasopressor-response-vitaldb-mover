#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
export PYTHONPATH="$PWD/local_deps${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/logs
case "${1:-}" in
 --prepare) "$PY" src/local_prepare.py ;;
 --gate) "$PY" src/local_recheck_gate.py ;;
 --analyze)
  "$PY" src/local_recheck_gate.py
  "$PY" src/local_extract_events.py --config config/mover_columns_local.yaml 2>&1 | tee outputs/logs/m1_console.log
  "$PY" src/local_model_and_external.py 2>&1 | tee outputs/logs/m2_console.log
  "$PY" src/local_report.py
  "$PY" src/local_finalize.py
  ;;
 --finalize) "$PY" src/local_report.py; "$PY" src/local_finalize.py ;;
 *) echo 'Usage: local_run.sh --prepare | --gate | --analyze | --finalize'; exit 1 ;;
esac
