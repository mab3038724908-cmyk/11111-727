#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT/src/td25a_robot_ui${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"$PYTHON_BIN" -m py_compile \
  "$ROOT/src/td25a_robot_ui/td25a_robot_ui/algorithms/free_space.py" \
  "$ROOT/src/td25a_robot_ui/td25a_robot_ui/algorithms/cleaning_mode.py" \
  "$ROOT/src/td25a_robot_ui/td25a_robot_ui/algorithms/grid_coverage.py" \
  "$ROOT/src/td25a_robot_ui/td25a_robot_ui/algorithms/path_manager.py" \
  "$ROOT/src/td25a_robot_ui/td25a_robot_ui/store/cleaned_area.py" \
  "$ROOT/tools/export_plan_contract.py"

"$PYTHON_BIN" -m pytest -q \
  "$ROOT/src/td25a_robot_ui/test/test_auto_clean_field.py" \
  "$ROOT/src/td25a_robot_ui/test/test_partitioned_coverage.py" \
  "$ROOT/src/td25a_robot_ui/test/test_path_manager.py" \
  "$ROOT/src/td25a_robot_ui/test/test_path_chunks.py" \
  "$ROOT/src/td25a_robot_ui/test/test_grid_coverage.py" \
  "$ROOT/src/td25a_robot_ui/test/test_theta_coverage_connectors.py" \
  "$ROOT/src/td25a_robot_ui/test/test_coverage_fixes_tonight.py" \
  "$ROOT/src/td25a_robot_ui/test/test_coverage_spacing.py" \
  "$ROOT/src/td25a_robot_ui/test/test_cleaned_area_tracker.py" \
  "$ROOT/tests/test_reference_cleaner_contract.py" \
  "$ROOT/tests/test_delivery_contract.py"

mkdir -p "$ROOT/reference_outputs"
"$PYTHON_BIN" "$ROOT/tools/benchmark_synthetic_coverage.py" \
  --random-count 2 --random-manual-count 1 --random-office-count 1 \
  --json "$ROOT/reference_outputs/generated_synthetic_metrics.json"

"$PYTHON_BIN" "$ROOT/tools/benchmark_partitioned_coverage.py" \
  --maps-dir "$ROOT/maps" --map 9 \
  --json "$ROOT/reference_outputs/generated_map9_metrics.json"

echo "Offline checks completed successfully."
