export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATASETS=("sst2" "poem_sentiment" "coin_flip" "addition" "edge_exist")

for dataset in "${DATASETS[@]}"; do
  python test_multitask.py \
    --dataset "$dataset" \
    --gradce --k 50 --model_name Qwen/Qwen2.5-3B-Instruct\
    --out_dir out/gradce_run \
    --is_quant --max_length 90 --filter 70
done