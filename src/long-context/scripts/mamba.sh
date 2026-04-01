#!/bin/bash

TRAIN_DATASETS="sst2"
TRAIN_K=50
TRAIN_DATASET_TAG="${TRAIN_DATASETS//,/_}"
PHASE1_CKPT="checkpoints/debug_phase1_${TRAIN_DATASET_TAG}_${TRAIN_K}.pt"
PHASE2_CKPT="checkpoints/debug_phase2_${TRAIN_DATASET_TAG}_${TRAIN_K}.pt"

# ============================================================
# Phase 1: L_KV only (coarse initialization)
# ============================================================

echo "=== Phase 1: KV-matching pretraining ==="
python mamba_debug.py \
  --dataset $TRAIN_DATASETS \
  --run_mode train \
  --k $TRAIN_K \
  --retrieval_split test \
  --train_split test \
  --eval_split dev \
  --epochs 3 \
  --num_virtual_tokens 16 \
  --sink_tokens 4 \
  --ssm_dim 512 \
  --num_groups 4 \
  --num_demo_sets 500 \
  --learning_rate 3e-4 \
  --loss_w_kv 1.0 \
  --loss_w_logit 0.0 \
  --loss_w_hid 0.0 \
  --kv_loss_type cosine \
  --train_batch_size 4 \
  --kd_temperature 2.0 \
  --grad_clip 1.0 \
  --log_every 20 \
  --save_sidecar_path "$PHASE1_CKPT"

# ============================================================
# Phase 2: L_logit + L_hid (no L_KV)
#   - L_logit: KL divergence on output logits
#   - L_hid: cosine distance on last-layer hidden states
# ============================================================

echo "=== Phase 2: Logit + hidden-state fine-tuning ==="
python mamba_debug.py \
  --dataset $TRAIN_DATASETS \
  --run_mode train \
  --k $TRAIN_K \
  --retrieval_split test \
  --train_split test \
  --eval_split dev \
  --epochs 3 \
  --num_virtual_tokens 16 \
  --sink_tokens 4 \
  --ssm_dim 512 \
  --num_groups 4 \
  --num_demo_sets 500 \
  --learning_rate 5e-5 \
  --loss_w_kv 0.0 \
  --loss_w_logit 1.0 \
  --loss_w_hid 1.0 \
  --kv_loss_type cosine \
  --train_batch_size 4 \
  --kd_temperature 2.0 \
  --grad_clip 1.0 \
  --log_every 20 \
  --load_sidecar_path "$PHASE1_CKPT" \
  --save_sidecar_path "$PHASE2_CKPT"

# ============================================================
# Eval
# ============================================================

echo "=== Eval ==="
IFS=',' read -r -a DATASET_LIST <<< "$TRAIN_DATASETS"
for dataset in "${DATASET_LIST[@]}"; do
  dataset="${dataset//[[:space:]]/}"
  echo "--- Evaluating: $dataset ---"
  python mamba_debug.py \
    --dataset "$dataset" \
    --run_mode eval \
    --retrieval_split test \
    --eval_split dev \
    --num_virtual_tokens 16 \
    --sink_tokens 4 \
    --ssm_dim 512 \
    --num_groups 4 \
    --num_eval_demo_sets 2 \
    --load_sidecar_path "$PHASE2_CKPT" \
    --k $TRAIN_K
done