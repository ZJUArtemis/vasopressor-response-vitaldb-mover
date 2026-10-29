#!/bin/bash
# MOVER host extraction. Read-only; no modelling. Writes only into ./outputs/.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
mkdir -p outputs/logs

case "${1:-}" in
  --inspect)
    "$PY" src/inspect_columns.py config/mover_columns.yaml
    ;;
  --test)
    "$PY" src/make_fixture.py
    "$PY" src/extract_events.py --config fixtures/mover_columns.yaml --max_cases 80
    echo
    echo "SMOKE TEST OK. Clean up before the real run:"
    echo "  rm -rf outputs/* fixtures/*"
    ;;
  --extract)
    "$PY" src/extract_events.py --config config/mover_columns.yaml \
        --previous_events "${PREVIOUS_EVENTS:-}" 2>&1 | tee outputs/logs/extract_console.log
    ;;
  --package)
    "$PY" src/package_return.py
    ;;
  *)
    echo "Usage: run.sh --inspect | --test | --extract | --package"
    exit 1
    ;;
esac
