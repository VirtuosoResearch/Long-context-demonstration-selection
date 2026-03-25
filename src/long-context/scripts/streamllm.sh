#!/bin/bash
set -euo pipefail

DATASETS=("sst2" "poem_sentiment" "coin_flip" "addition")

for dataset in "${DATASETS[@]}"; do
    python streaming_llm_baseline.py \
        --dataset "$dataset" \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --k 50 --sink_tokens 4 \
        --recent_tokens_list 32,64,128 \
        --eval_split dev --retrieval_split test \
        --flops --flops_profile_samples 10
done