#!/usr/bin/env bash

set -euo pipefail

# Public, git-safe runner for the MMTR benchmark pipeline.
#
# Expected dataset layout:
#   <DATASET_DIR>/
#     MMTR.jsonl
#     images/
#
# Usage:
#   DATASET_DIR=/path/to/mmtr_dataset bash run_mmtr_pipeline.sh
#   DATASET_DIR=/path/to/mmtr_dataset bash run_mmtr_pipeline.sh test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-.}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${DATASET_DIR}/mmtr_eval_outputs}"

INPUT_JSONL="${INPUT_JSONL:-${DATASET_DIR}/MMTR.jsonl}"
DATASET_ROOT="${DATASET_ROOT:-${DATASET_DIR}}"
OUT_DIR="${OUTPUT_BASE_DIR}/full"
EXTRA_ARGS=()

AZURE_API_KEY="${AZURE_OPENAI_API_KEY:-}"
AZURE_BASE_URL="${AZURE_OPENAI_BASE_URL:-}"
AZURE_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
INFER_MODEL_NAME="${INFER_MODEL_NAME:-gpt-5}"
INFER_REASONING_EFFORT="${INFER_REASONING_EFFORT:-high}"
INFER_WORKERS="${INFER_WORKERS:-8}"

CLEANER_BACKEND="${CLEANER_BACKEND:-rule}"
CLEANER_MODEL_NAME="${CLEANER_MODEL_NAME:-gpt-5-mini}"
CLEANER_API_KEY="${CLEANER_API_KEY:-${AZURE_OPENAI_API_KEY:-}}"
CLEANER_BASE_URL="${CLEANER_BASE_URL:-${AZURE_OPENAI_BASE_URL:-}}"
CLEANER_API_VERSION="${CLEANER_API_VERSION:-${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}}"
CLEANER_WORKERS="${CLEANER_WORKERS:-8}"

QWEN_EMBED_MODEL="${QWEN_EMBED_MODEL:-}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-256}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-0}"
LLM_JUDGE_LEVELS="${LLM_JUDGE_LEVELS:-L2,L3,L4}"
LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-}"
LLM_JUDGE_BASE_URL="${LLM_JUDGE_BASE_URL:-}"
LLM_JUDGE_MODEL_NAME="${LLM_JUDGE_MODEL_NAME:-}"

if [[ "${MODE}" == "test" ]]; then
  OUT_DIR="${OUTPUT_BASE_DIR}/test"
  EXTRA_ARGS+=(--test_mode)
fi

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "Input jsonl not found: ${INPUT_JSONL}" >&2
  exit 1
fi

if [[ ! -d "${DATASET_ROOT}/images" ]]; then
  echo "Expected images directory not found: ${DATASET_ROOT}/images" >&2
  exit 1
fi

if [[ -z "${AZURE_API_KEY}" ]]; then
  echo "Missing Azure API key. Set AZURE_OPENAI_API_KEY." >&2
  exit 1
fi

if [[ -z "${AZURE_BASE_URL}" ]]; then
  echo "Missing Azure base URL. Set AZURE_OPENAI_BASE_URL." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

STEP1_ARGS=(
  --input_jsonl "${INPUT_JSONL}"
  --output_jsonl "${OUT_DIR}/mmtr_infer.jsonl"
  --dataset_root "${DATASET_ROOT}"
  --model_name "${INFER_MODEL_NAME}"
  --reasoning_effort "${INFER_REASONING_EFFORT}"
  --api_key "${AZURE_API_KEY}"
  --base_url "${AZURE_BASE_URL}"
  --api_version "${AZURE_API_VERSION}"
  --workers "${INFER_WORKERS}"
)

STEP2_ARGS=(
  --input_jsonl "${OUT_DIR}/mmtr_infer.jsonl"
  --output_jsonl "${OUT_DIR}/mmtr_clean.jsonl"
  --cleaner_backend "${CLEANER_BACKEND}"
  --model_name "${CLEANER_MODEL_NAME}"
  --api_key "${CLEANER_API_KEY}"
  --base_url "${CLEANER_BASE_URL}"
  --api_version "${CLEANER_API_VERSION}"
  --workers "${CLEANER_WORKERS}"
)

STEP3_ARGS=(
  --input_jsonl "${OUT_DIR}/mmtr_clean.jsonl"
  --output_jsonl "${OUT_DIR}/mmtr_eval.jsonl"
  --num_workers "${EVAL_NUM_WORKERS}"
  --qwen_batch_size "${QWEN_BATCH_SIZE}"
  --llm_judge_levels "${LLM_JUDGE_LEVELS}"
)

if [[ -n "${QWEN_EMBED_MODEL}" ]]; then
  STEP3_ARGS+=(--qwen_model_name "${QWEN_EMBED_MODEL}")
else
  STEP3_ARGS+=(--disable_qwen_embed)
fi

if [[ "${ENABLE_LLM_JUDGE}" == "1" ]]; then
  STEP3_ARGS+=(
    --llm_api_key "${LLM_JUDGE_API_KEY}"
    --llm_base_url "${LLM_JUDGE_BASE_URL}"
    --llm_model_name "${LLM_JUDGE_MODEL_NAME}"
  )
else
  STEP3_ARGS+=(--disable_llm_judge)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/step1_infer_mmtr_azure.py" \
  "${STEP1_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/step2_clean_mmtr_prediction.py" \
  "${STEP2_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/step3_eval_mmtr_llm.py" \
  "${STEP3_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
