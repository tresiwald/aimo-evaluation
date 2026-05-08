#!/usr/bin/env bash
set -euo pipefail

export OPENAI_COMPATIBLE_BASE_URL="https://numerate-stainable-washing.ngrok-free.dev/v1"
export OPENAI_COMPATIBLE_API_KEY="5faa6cf84803ef805d4f5824733c0c5d157400c12f0906118a4c83291d8c497a"

MODEL="gpt-oss-120b-aimo3-gguf"
OUT_DIR="data"
VARIANTS=5
MAX_CONCURRENCY=1
MAX_RETRIES=2
RETRY_SLEEP=5

DATASETS=(
  "data/hendrycks-math-test-reference-100.jsonl"
  "data/hendrycks-math-train-reference-500.jsonl"
)

PROMPTS=(
  "prompts/distract.txt"
  "prompts/domain.txt"
  "prompts/rename.txt"
  "prompts/rephrase.txt"
  "prompts/typos.txt"
)

for dataset in "${DATASETS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do
    python3 robustness-analyses/main.py augment \
      "$dataset" \
      "$prompt" \
      "$OUT_DIR" \
      --provider openai-compatible \
      --api-model "$MODEL" \
      --n-variants "$VARIANTS" \
      --max-concurrency "$MAX_CONCURRENCY" \
      --max-retries "$MAX_RETRIES" \
      --retry-sleep-secs "$RETRY_SLEEP"
  done
done
