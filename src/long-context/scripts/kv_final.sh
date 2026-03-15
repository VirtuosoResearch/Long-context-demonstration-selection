python test_multitask.py \
  --dataset poem_sentiment,addition,coin_flip,cr \
  --kv_final --k 20 --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --out_dir out/multitask_run \
  --is_quant \
  --kv_final_lambda 0.9  --kv_final_subset_multiplier 1.5 --filter 100 --max_length 256

