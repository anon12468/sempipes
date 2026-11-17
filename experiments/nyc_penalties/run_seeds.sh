#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="experiments/nyc_penalties/logs"
mkdir -p "$LOG_DIR"

SEEDS="${SEEDS:-0 1 2 3 4}"
MODEL="${MODEL:-gemini/gemini-2.5-flash}"  # LiteLLM model name
MODEL_TAG="${MODEL//\//_}"  # "/" is not allowed in log file names
TEMPERATURE="${TEMPERATURE:-0.0}"

for seed in $SEEDS; do
  log="$LOG_DIR/nyc_penalties_${MODEL_TAG}_seed_${seed}.log"
  echo "Starting seed=$seed model=$MODEL temperature=$TEMPERATURE -> $log"
  nohup poetry run python -u -m experiments.nyc_penalties.sempipes \
    --seed "$seed" --llm "$MODEL" --temperature "$TEMPERATURE" \
    > "$log" 2>&1 &
done

echo "Launched $(echo "$SEEDS" | wc -w) jobs. Tail logs with:"
echo "  tail -f $LOG_DIR/nyc_penalties_${MODEL_TAG}_seed_*.log"
