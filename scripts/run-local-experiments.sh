#!/usr/bin/env bash
#
# Reproducible local experiment suite. Builds the jar once, runs each
# experiment, and writes per-experiment logs plus a combined summary of the
# machine-readable RESULT / SCALING lines to the output directory.
#
#   ./scripts/run-local-experiments.sh [output-dir]
#
# Environment overrides:
#   MASTER=local[8]   Spark master (default local[4])
#   DATA=data/scaling Root holding parquet_10k / parquet_100k / parquet_1m
#
# Experiments whose input data is missing are skipped with a notice rather
# than failing the run, so a partial data setup still produces useful output.
# See README.md for how to build the Parquet inputs.

set -uo pipefail

cd "$(dirname "$0")/.."

RESULTS_DIR=${1:-results}
MASTER=${MASTER:-local[4]}
DATA=${DATA:-data/scaling}
JAR=ensemble.jar

mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.txt"
: > "$SUMMARY"

echo "Building jar..."
if ! make jar > "$RESULTS_DIR/build.log" 2>&1; then
  echo "Build failed - see $RESULTS_DIR/build.log" >&2
  exit 1
fi
echo "Build ok."
echo

# run <name> <spark-submit args...>
run() {
  local name=$1; shift
  local log="$RESULTS_DIR/${name}.log"
  local start elapsed
  echo "==> ${name}"
  start=$(date +%s)
  if spark-submit "$@" > "$log" 2>&1; then
    elapsed=$(( $(date +%s) - start ))
    echo "    ok (${elapsed}s) -> ${log}"
  else
    elapsed=$(( $(date +%s) - start ))
    echo "    FAILED (${elapsed}s) -> ${log}" >&2
  fi
  # Both markers are printed by the drivers themselves; absence is not an error
  # (component smoke tests legitimately print neither).
  grep -hE '^(RESULT|SCALING)' "$log" >> "$SUMMARY" 2>/dev/null
  return 0
}

# skip_unless_data <dir> ; returns 1 (and warns) when the Parquet pair is absent
have_data() {
  if [ -d "$1/train" ] && [ -d "$1/test" ]; then
    return 0
  fi
  echo "==> skipping: $1 not found (see README.md for data setup)"
  return 1
}

echo "### Toolchain check"
run toolchain --class smoketest.ToolchainSmokeTest --master "$MASTER" "$JAR"
echo

echo "### Scaling harness (12-config sweep at each input size)"
for size in 10k 100k 1m; do
  d="$DATA/parquet_${size}"
  if have_data "$d"; then
    run "scaling_${size}" --class driver.ScalingHarnessDriver --master "$MASTER" \
      --driver-memory 4g "$JAR" "$d/train" "$d/test" "$size"
  fi
done
echo

# The remaining experiments use the 100K subset: large enough that results are
# not dominated by small-sample noise, small enough to rerun quickly.
BASE="$DATA/parquet_100k"
if have_data "$BASE"; then
  echo "### Single ensemble configuration"
  run experiment --class driver.EnsembleExperimentDriver --master "$MASTER" \
    "$JAR" "$BASE/train" "$BASE/test" 10
  echo

  echo "### Hyperparameter search (naive parallel + nested-prefix)"
  run hpsearch --class smoketest.HyperparameterSearchSmokeTest --master "$MASTER" \
    "$JAR" "$BASE/train" "$BASE/test"
  echo

  echo "### Design levers (sampling strategy x aggregation strategy)"
  run levers --class smoketest.DesignLeverSmokeTest --master "$MASTER" \
    "$JAR" "$BASE/train" "$BASE/test" 10
  echo

  echo "### MLlib baseline (independent comparison)"
  run mllib_baseline --class mllibbaseline.MLlibBaselineDriver --master "$MASTER" \
    --driver-memory 4g "$JAR" "$BASE/train" "$BASE/test"
  echo
fi

echo "======================================================================"
echo "Logs:    $RESULTS_DIR/"
echo "Summary: $SUMMARY"
echo "======================================================================"
cat "$SUMMARY"
