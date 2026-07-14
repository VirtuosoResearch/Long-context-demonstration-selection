#!/bin/bash
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Defaults (can be overridden by env vars when invoking this script)
MODEL_NAME_GSM8K="${MODEL_NAME_GSM8K:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_NAME_OTHERS="${MODEL_NAME_OTHERS:-Qwen/Qwen2.5-3B-Instruct}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
IS_QUANT="${IS_QUANT:-0}"   # 1 => enable --is_quant

NUM_EVAL_DEMO_SETS="${NUM_EVAL_DEMO_SETS:-1}"
MAX_DEMO_TOKENS="${MAX_DEMO_TOKENS:-1536}"

# Noisy self-calibration defaults
CALIB_STEPS="${CALIB_STEPS:-100}"
CALIB_LR="${CALIB_LR:-1e-2}"
CALIB_NOISE_STD="${CALIB_NOISE_STD:-1e-2}"
CALIB_BWD_FACTOR="${CALIB_BWD_FACTOR:-3.0}"

# I2CL coefficient init defaults
LAMBDA_A="${LAMBDA_A:-0.1}"
BETA_A="${BETA_A:-1.0}"
LAMBDA_M="${LAMBDA_M:-0.1}"
BETA_M="${BETA_M:-1.0}"

# Datasets requested by user: mmlu, modular addition (addition), gsm8k
EVAL_DATASETS=(mmlu addition gsm8k)

echo "============================================================"
echo "I2CL eval (mmlu / addition / gsm8k)"
echo "SEED=${SEED}, DEVICE=${DEVICE}, DTYPE=${DTYPE}, IS_QUANT=${IS_QUANT}"
echo "num_eval_demo_sets=${NUM_EVAL_DEMO_SETS}, max_demo_tokens=${MAX_DEMO_TOKENS}"
echo "calibration: steps=${CALIB_STEPS}, lr=${CALIB_LR}, noise_std=${CALIB_NOISE_STD}"
echo "coeff init: lambda_a=${LAMBDA_A}, beta_a=${BETA_A}, lambda_m=${LAMBDA_M}, beta_m=${BETA_M}"
echo "============================================================"

for DATASET in "${EVAL_DATASETS[@]}"; do
  if [[ "${DATASET}" == "gsm8k" ]]; then
    MODEL_NAME="${MODEL_NAME_GSM8K}"
    K="${K_GSM8K:-10}"
    RETRIEVAL_SPLIT="${RETRIEVAL_SPLIT_GSM8K:-train}"
    EVAL_SPLIT="${EVAL_SPLIT_GSM8K:-test}"
  else
    MODEL_NAME="${MODEL_NAME_OTHERS}"
    K="${K_OTHERS:-50}"
    RETRIEVAL_SPLIT="${RETRIEVAL_SPLIT_OTHERS:-test}"
    EVAL_SPLIT="${EVAL_SPLIT_OTHERS:-dev}"
  fi

  echo
  echo "=== I2CL eval: dataset=${DATASET}, model=${MODEL_NAME}, k=${K} ==="
  echo "retrieval_split=${RETRIEVAL_SPLIT}, eval_split=${EVAL_SPLIT}"

  EXTRA_ARGS=()
  if [[ "${IS_QUANT}" == "1" ]]; then
    EXTRA_ARGS+=(--is_quant)
  fi

  python i2cl_baseline.py \
    --dataset "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --k "${K}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --retrieval_split "${RETRIEVAL_SPLIT}" \
    --eval_split "${EVAL_SPLIT}" \
    --num_eval_demo_sets "${NUM_EVAL_DEMO_SETS}" \
    --max_demo_tokens "${MAX_DEMO_TOKENS}" \
    --lambda_a "${LAMBDA_A}" \
    --beta_a "${BETA_A}" \
    --lambda_m "${LAMBDA_M}" \
    --beta_m "${BETA_M}" \
    --calibration_steps "${CALIB_STEPS}" \
    --calibration_lr "${CALIB_LR}" \
    --calibration_noise_std "${CALIB_NOISE_STD}" \
    --calibration_backward_factor "${CALIB_BWD_FACTOR}" \
    "${EXTRA_ARGS[@]}"
done

echo
echo "All I2CL eval runs finished."
