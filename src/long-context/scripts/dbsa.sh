#!/bin/bash
set -euo pipefail

DATASETS=("sst2" "poem_sentiment" "addition" "coin_flip" "edge_exist")

for dataset in "${DATASETS[@]}"; do 
python dbsa_baseline.py \
      --dataset "$dataset" \
      --model_name Qwen/Qwen2.5-1.5B-Instruct \
      --k 50 --block_size 5 --n_prev_blocks 2 --retrieval_ratio 0.3 \
      --eval_split dev --retrieval_split test
done


for dataset in "${DATASETS[@]}"; do 
python dbsa_baseline.py \
      --dataset "$dataset" \
      --model_name Qwen/Qwen2.5-3B-Instruct \
      --k 50 --block_size 5 --n_prev_blocks 2 --retrieval_ratio 0.3 \
      --eval_split dev --retrieval_split test
done



bash -x scripts/mamba_debug_all.sh