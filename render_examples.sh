#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT/src/td25a_robot_ui${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/reference_outputs"

"$PYTHON_BIN" "$ROOT/tools/render_partitioned_planner.py" \
  "$ROOT/maps/9.yaml" "$ROOT/reference_outputs/map9_current_profile.png" \
  --seed -1.73 -3.34 --label "map9 current sparse_graph profile"

"$PYTHON_BIN" "$ROOT/tools/render_synthetic_preview.py" \
  four_rooms_corridor \
  "$ROOT/reference_outputs/four_rooms_corridor.png"

"$PYTHON_BIN" "$ROOT/tools/export_plan_contract.py" \
  "$ROOT/maps/9.yaml" \
  "$ROOT/reference_outputs/map9_plan_summary.json" \
  --summary-only

echo "Reference outputs written to $ROOT/reference_outputs"

