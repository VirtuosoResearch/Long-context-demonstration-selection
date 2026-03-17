# #!/bin/bash
# set -euo pipefail

# # ============================================================
# # Phase 1: KV-matching only (fast convergence, strong signal)
# #   - loss_w_logit=0 means we ONLY use the direct KV matching loss
# #   - This teaches the SSM the right "shape" of KV to produce
# #   - 1-2 epochs is usually enough for this phase
# # ============================================================

# echo "=== Phase 1: KV-matching pretraining ==="
# python mamba_inference_multi.py \
#   --dataset sst2,cr,coin_flip \
#   --run_mode train \
#   --train_split dev \
#   --retrieval_split dev \
#   --eval_split dev \
#   --epochs 2 \
#   --num_virtual_tokens 16 \
#   --sink_tokens 4 \
#   --ssm_dim 512 \
#   --num_groups 4 \
#   --num_demo_sets 500 \
#   --learning_rate 3e-4 \
#   --loss_w_kv 1.0 \
#   --loss_w_logit 0.0 \
#   --kv_loss_type cosine \
#   --train_batch_size 4 \
#   --kd_temperature 2.0 \
#   --grad_clip 1.0 \
#   --log_every 20 \
#   --save_sidecar_path checkpoints/sidecar_phase1.pt

# # ============================================================
# # Phase 2: End-to-end fine-tuning (KV + logit distillation)
# #   - Load phase-1 checkpoint, then add logit loss
# #   - Lower LR for fine-tuning
# #   - This teaches the SSM to produce KV that works well for
# #     actual downstream QA, not just matching teacher KV shape
# # ============================================================

# echo "=== Phase 2: End-to-end fine-tuning ==="
# python mamba_inference_multi.py \
#   --dataset sst2,cr,coin_flip \
#   --run_mode train_compare_loss \
#   --train_split dev \
#   --retrieval_split dev \
#   --eval_split dev \
#   --epochs 1 \
#   --num_virtual_tokens 16 \
#   --sink_tokens 4 \
#   --ssm_dim 512 \
#   --num_groups 4 \
#   --num_demo_sets 500 \
#   --learning_rate 5e-5 \
#   --loss_w_kv 0.3 \
#   --loss_w_logit 1.0 \
#   --kv_loss_type cosine \
#   --train_batch_size 4 \
#   --kd_temperature 2.0 \
#   --grad_clip 1.0 \
#   --log_every 20 \
#   --load_sidecar_path checkpoints/sidecar_phase1.pt \
#   --save_sidecar_path checkpoints/sidecar_phase2.pt

# # ============================================================
# # Eval: accuracy + loss comparison
# # ============================================================

# echo "=== Final evaluation ==="
# python mamba_inference_multi.py \
#   --dataset sst2,cr,coin_flip \
#   --run_mode eval \
#   --retrieval_split dev \
#   --eval_split dev \
#   --num_virtual_tokens 16 \
#   --sink_tokens 4 \
#   --ssm_dim 512 \
#   --num_groups 4 \
#   --load_sidecar_path checkpoints/sidecar_phase2.pt


# # python mamba_inference.py \
# #   --dataset sst2,cr,coin_flip \
# #   --run_mode train_compare_loss \
# #   --train_split dev \
# #   --retrieval_split dev \
# #   --eval_split dev \
# #   --epochs 1 \
# #   --num_virtual_tokens 16 \
# #   --sink_tokens 4

python mamba_inference_multi.py \
  --dataset sst2,cr,coin_flip \
  --run_mode eval \
  --retrieval_split dev \
  --eval_split dev \
  --num_virtual_tokens 16 \
  --sink_tokens 4 \
  --ssm_dim 512 \
  --num_groups 4 \
  --load_sidecar_path checkpoints/sidecar_phase2.pt