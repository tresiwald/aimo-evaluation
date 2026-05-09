#!/bin/bash
#SBATCH --job-name=hendrycks-math-predict
#SBATCH --output=hendrycks_math_predict_o%A_%a
#SBATCH --partition=gpu-single
#SBATCH --tasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --mem=80gb
#SBATCH --gres=gpu:A40:1
#SBATCH --array=0-11

set -euo pipefail

module load devel/miniforge
source ./.venv/bin/activate

export OPENAI_COMPATIBLE_BASE_URL="${OPENAI_COMPATIBLE_BASE_URL:-https://numerate-stainable-washing.ngrok-free.dev/v1}"
export OPENAI_COMPATIBLE_API_KEY="${OPENAI_COMPATIBLE_API_KEY:-5faa6cf84803ef805d4f5824733c0c5d157400c12f0906118a4c83291d8c497a}"

PROVIDER="${PROVIDER:-openai-compatible}"
MODEL_NAME="${MODEL_NAME:-gpt-oss-120b-aimo3-gguf}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-predictions}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_SLEEP_SECS="${RETRY_SLEEP_SECS:-5}"
N_REPEATS="${N_REPEATS:-1}"
ON_FILE_EXISTS="${ON_FILE_EXISTS:-fill-missing}"

INPUT_FILES=(
  "data/hendrycks-math-test-reference-100.jsonl"
  "data/hendrycks-math-test-reference-100___distract=gpt-oss-120b-aimo3-gguf:low.jsonl"
  "data/hendrycks-math-test-reference-100___domain=gpt-oss-120b-aimo3-gguf:low.jsonl"
  "data/hendrycks-math-test-reference-100___rename=gpt-oss-120b-aimo3-gguf:low.jsonl"
  "data/hendrycks-math-test-reference-100___rephrase=gpt-oss-120b-aimo3-gguf:low.jsonl"
  "data/hendrycks-math-test-reference-100___typos=gpt-oss-120b-aimo3-gguf:low.jsonl"
  "data/hendrycks-math-train-reference-500.jsonl"
  "data/hendrycks-math-train-reference-500___distract=gpt-oss-120b-aimo3-gguf:low.jsonl"
  "data/hendrycks-math-train-reference-500___domain=gpt-oss-120b-aimo3-gguf:low.jsonl"
  "data/hendrycks-math-train-reference-500___rename=gpt-oss-120b-aimo3-gguf:low.jsonl"
  "data/hendrycks-math-train-reference-500___rephrase=gpt-oss-120b-aimo3-gguf:low.jsonl"
  "data/hendrycks-math-train-reference-500___typos=gpt-oss-120b-aimo3-gguf:low.jsonl"
)

INPUT_FILE="${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Missing input file: $INPUT_FILE" >&2
  exit 1
fi

mkdir -p "$PREDICTIONS_DIR"

python3 robustness-analyses/main.py predict \
  "$INPUT_FILE" \
  "$PREDICTIONS_DIR" \
  --provider "$PROVIDER" \
  --api-model "$MODEL_NAME" \
  --n-repeats "$N_REPEATS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --max-retries "$MAX_RETRIES" \
  --retry-sleep-secs "$RETRY_SLEEP_SECS" \
  --on-file-exists "$ON_FILE_EXISTS"
