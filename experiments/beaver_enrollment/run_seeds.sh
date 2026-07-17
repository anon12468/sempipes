#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-40}"

LOG_DIR="experiments/beaver_enrollment/logs"
mkdir -p "$LOG_DIR"

SEEDS="${SEEDS:-0 1 2 3 4}"

for seed in $SEEDS; do
  log="$LOG_DIR/enrollment_seed_${seed}.log"
  echo "Starting seed=$seed -> $log"
  nohup poetry run python -u -m experiments.beaver_enrollment.pipeline --seed "$seed" \
    > "$log" 2>&1 &
done

echo "Launched $(echo $SEEDS | wc -w) jobs. Tail logs with:"
echo "  tail -f $LOG_DIR/enrollment_seed_*.log"
