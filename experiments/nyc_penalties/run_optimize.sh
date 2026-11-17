#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="experiments/nyc_penalties/logs"
mkdir -p "$LOG_DIR"

SEEDS="${SEEDS:-0 1 2 3 4}"
MODEL="${MODEL:-gemini/gemini-2.5-flash}"  # LiteLLM model name
MODEL_TAG="${MODEL//\//_}"  # "/" is not allowed in log file names
STRATEGY="${STRATEGY:-epsilon_greedy}"
EXPLORATION_PARAM="${EXPLORATION_PARAM:-0.5}"

if [[ "$STRATEGY" == "epsilon_greedy" ]]; then
  STRATEGY_TAG="eps${EXPLORATION_PARAM}"
elif [[ "$STRATEGY" == "round_robin" ]]; then
  STRATEGY_TAG="rr"
else
  STRATEGY_TAG="${STRATEGY}_${EXPLORATION_PARAM}"
fi

for seed in $SEEDS; do
  log="$LOG_DIR/nyc_penalties_${MODEL_TAG}_mct_${STRATEGY_TAG}_${seed}.log"
  echo "Starting optimize seed=$seed model=$MODEL strategy=$STRATEGY exploration=$EXPLORATION_PARAM -> $log"
  nohup poetry run python -u -m experiments.nyc_penalties.minibench \
    nyc_penalties "$MODEL" mct_search "$seed" "$STRATEGY" "$EXPLORATION_PARAM" \
    > "$log" 2>&1 &
done

echo "Launched $(echo "$SEEDS" | wc -w) optimization jobs."
echo "  tail -f $LOG_DIR/nyc_penalties_${MODEL_TAG}_mct_${STRATEGY_TAG}_*.log"
