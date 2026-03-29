#!/bin/bash
# set -euo pipefail
# "sst2" "poem_sentiment" "coin_flip" "addition" 
DATASETS=("sst2" "poem_sentiment" "coin_flip" "addition" "edge_exist")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for dataset in "${DATASETS[@]}"; do
    python duo_attention_baseline.py \
        --dataset "$dataset" \
        --model_name Qwen/Qwen2.5-3B-Instruct \
        --k 50 --sink_tokens 4 \
        --recent_tokens 64 \
        --dtype auto \
        --gate_max_seq_len 768 \
        --gate_num_passkeys 4 \
        --gate_train_steps 400 \
        --gate_attn_chunk_size 128 \
        --run_mode identify_eval
done
