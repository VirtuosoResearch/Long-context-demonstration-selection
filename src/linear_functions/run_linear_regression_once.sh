#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_linear_regression_once.sh               # quick sanity run (recommended)
#   bash run_linear_regression_once.sh quick         # same as default
#   bash run_linear_regression_once.sh full          # full linear_regression config
#   bash run_linear_regression_once.sh eval_pretrained
#
# Notes:
# - quick mode uses --test_run True so it skips wandb and runs only 100 steps.
# - run from anywhere; script will cd into src/ automatically.

MODE="${1:-quick}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"

cd "${SRC_DIR}"

echo "[INFO] Working dir: ${SRC_DIR}"
echo "[INFO] Mode: ${MODE}"

if [[ "${MODE}" == "quick" ]]; then
  echo "[INFO] Running quick linear regression sanity train..."
  python train.py --config conf/linear_regression.yaml --test_run True
  echo "[DONE] quick run finished."
elif [[ "${MODE}" == "full" ]]; then
  echo "[INFO] Running full linear regression training..."
  echo "[WARN] This uses wandb settings from conf/wandb.yaml and can take a long time."
  python train.py --config conf/linear_regression.yaml
  echo "[DONE] full run finished."
elif [[ "${MODE}" == "eval_pretrained" ]]; then
  echo "[INFO] Evaluating pretrained checkpoints under ../models ..."
  python eval.py ../models
  echo "[DONE] pretrained eval finished."
else
  echo "[ERROR] Unknown mode: ${MODE}"
  echo "Valid modes: quick | full | eval_pretrained"
  exit 1
fi
