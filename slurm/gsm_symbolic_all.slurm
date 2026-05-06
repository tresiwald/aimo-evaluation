#!/bin/bash
#SBATCH --job-name=gsm-symbolic-all
#SBATCH --output=gsm_symbolic_all_o%j
#SBATCH --partition=gpu-single
#SBATCH --tasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --mem=80gb
#SBATCH --gres=gpu:A40:1

set -euo pipefail

module load devel/miniforge
source ./.venv/bin/activate

# Optional one-time sync if the environment on the cluster is stale.
# ~/.local/bin/uv sync

# Provider/model configuration.
PROVIDER="${PROVIDER:-huggingface-local}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-0528-Qwen3-8B}"

# Set MODE=smoke for a small validation run.
MODE="${MODE:-full}"

REFERENCE_DATA="data/gsm-symbolic-reference.jsonl"
PERMUTATION_DATA="data/gsm-symbolic-permutations.jsonl"
PREDICTIONS_DIR="predictions"

COMMON_ARGS=(
  --provider "$PROVIDER"
  --api-model "$MODEL_NAME"
  --n-repeats 1
  --max-concurrency 1
  --max-retries 2
  --retry-sleep-secs 5
)

SAFE_MODEL_NAME="${MODEL_NAME//\//__}"
REFERENCE_PRED="predictions/gsm-symbolic-reference___eval=${SAFE_MODEL_NAME}:low.jsonl"
PERMUTATION_PRED="predictions/gsm-symbolic-permutations___eval=${SAFE_MODEL_NAME}:low.jsonl"

if [[ "$MODE" == "smoke" ]]; then
  REFERENCE_EXTRA=(--row-offset 0 --max-rows 5 --output-suffix smoke)
  PERMUTATION_EXTRA=(--row-offset 0 --max-rows 20 --output-suffix smoke)
  REFERENCE_PRED="predictions/gsm-symbolic-reference___eval=${SAFE_MODEL_NAME}:low___smoke.jsonl"
  PERMUTATION_PRED="predictions/gsm-symbolic-permutations___eval=${SAFE_MODEL_NAME}:low___smoke.jsonl"
else
  REFERENCE_EXTRA=()
  PERMUTATION_EXTRA=()
fi

mkdir -p "$PREDICTIONS_DIR"

python robustness-analyses/main.py predict \
  "$REFERENCE_DATA" \
  "$PREDICTIONS_DIR" \
  "${COMMON_ARGS[@]}" \
  "${REFERENCE_EXTRA[@]}"

python robustness-analyses/main.py predict \
  "$PERMUTATION_DATA" \
  "$PREDICTIONS_DIR" \
  "${COMMON_ARGS[@]}" \
  "${PERMUTATION_EXTRA[@]}"

python robustness-analyses/main.py eval \
  "$PERMUTATION_PRED" \
  --base-pred-file "$REFERENCE_PRED"
