#!/usr/bin/env bash
# Run PAG-style SpecPrefill with plain HF Transformers prefill only.
# No SGLang/vLLM/TRT-LLM server and no token generation.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/specprefill}"
BENCH="${ROOT_DIR}/benchmark/run_vanilla_prefill.py"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TARGET_MODEL_ID="${TARGET_MODEL_ID:-/workspace/models/openai__gpt-oss-120b}"
export TARGET_TOKENIZER_ID="${TARGET_TOKENIZER_ID:-${TARGET_MODEL_ID}}"
export SCORER_MODEL_ID="${SCORER_MODEL_ID:-/workspace/models/openai__gpt-oss-20b}"
export SCORER_DEVICE="${SCORER_DEVICE:-cuda:3}"
export TARGET_DEVICE_MAP="${TARGET_DEVICE_MAP:-auto}"
export TARGET_DTYPE="${TARGET_DTYPE:-auto}"
export SCORER_DTYPE="${SCORER_DTYPE:-auto}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
export TARGET_MAX_MEMORY="${TARGET_MAX_MEMORY:-0:55GiB,1:55GiB,2:55GiB,3:4GiB,cpu:500GiB}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/vanilla_prefill}"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/benchmark/requirements.txt"
fi

if [[ "${SCORER_DEVICE}" == cuda* ]]; then
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import accelerate
PY
  then
    echo "==> Installing accelerate for GPU scorer device_map"
    "${PYTHON_BIN}" -m pip install -U accelerate
  fi
elif [[ "${SCORER_MODEL_ID}" == *gpt-oss* || "${SCORER_MODEL_ID}" == *gpt_oss* ]]; then
  cat >&2 <<'EOF'
ERROR: GPT-OSS scorer cannot run on CPU in this vanilla path.
The official GPT-OSS checkpoints use MXFP4/Triton kernels that expect CUDA
tensors. Use SCORER_DEVICE=cuda:<id>, or switch to a CPU-friendly scorer such
as SCORER_MODEL_ID=Qwen/Qwen3-0.6B.
EOF
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

echo "==> Vanilla HF prefill only"
echo "==> Target: ${TARGET_MODEL_ID}"
echo "==> Scorer: ${SCORER_MODEL_ID} on ${SCORER_DEVICE}"
echo "==> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "==> TARGET_MAX_MEMORY=${TARGET_MAX_MEMORY}"

ARGS=(
  --mode "${MODE:-prefill}"
  --target-model "${TARGET_MODEL_ID}"
  --target-tokenizer "${TARGET_TOKENIZER_ID}"
  --target-device-map "${TARGET_DEVICE_MAP}"
  --target-max-memory "${TARGET_MAX_MEMORY}"
  --target-dtype "${TARGET_DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --scorer-model "${SCORER_MODEL_ID}"
  --scorer-device "${SCORER_DEVICE}"
  --scorer-dtype "${SCORER_DTYPE}"
  --score-window "${SCORE_WINDOW:-2048}"
  --chunk-tokens "${CHUNK_TOKENS:-128}"
  --anchor-tokens "${ANCHOR_TOKENS:-512}"
  --prompt-tokens "${PROMPT_TOKENS:-16000}"
  --prefill-side "${PREFILL_SIDE:-both}"
  --keep-rates "${KEEP_RATES:-0.3}"
  --measurements "${MEASUREMENTS:-1}"
  --warmups "${WARMUPS:-0}"
  --quality-source "${QUALITY_SOURCE:-longbench}"
  --dataset-name "${DATASET_NAME:-THUDM/LongBench-v2}"
  --dataset-split "${DATASET_SPLIT:-test}"
  --quality-n "${QUALITY_N:-20}"
  --quality-prompt-tokens "${QUALITY_PROMPT_TOKENS:-16000}"
  --quality-keep-rates "${QUALITY_KEEP_RATES:-0.3,0.5}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi
if [[ -n "${QUALITY_JSONL:-}" ]]; then
  ARGS+=(--quality-jsonl "${QUALITY_JSONL}")
fi
if [[ "${SKIP_QUALITY_TARGET_PREFILL:-0}" == "1" ]]; then
  ARGS+=(--skip-quality-target-prefill)
fi
if [[ "${INCLUDE_LM_HEAD:-0}" == "1" ]]; then
  ARGS+=(--include-lm-head)
fi

"${PYTHON_BIN}" "${BENCH}" "${ARGS[@]}"

if [[ "${SAVE_RESULTS_STACK:-1}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/benchmark/save_vanilla_results_stack.py" \
    --results-dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/RESULTS.md" \
    --json-output "${OUTPUT_DIR}/RESULTS.json"
fi
