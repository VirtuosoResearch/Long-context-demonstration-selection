

# python test_multitask.py \
#   --dataset poem_sentiment,addition,coin_flip \
#   --kv_final --k 5 --model_name Qwen/Qwen3-4B-Instruct-2507\
#   --out_dir out/multitask_run \
#   --is_quant \
#   --kv_final_lambda 0.9  --kv_final_subset_multiplier 1.5 --filter 130 --max_length 128

# python test_multitask.py \
#   --dataset sst2,cr \
#   --kv_final --k 5 --model_name Qwen/Qwen3-4B-Instruct-2507\
#   --out_dir out/multitask_run \
#   --is_quant \
#   --kv_final_lambda 0.9  --kv_final_subset_multiplier 1.5 --filter 130 --max_length 128

# python test_multitask.py \
#   --dataset poem_sentiment,addition,coin_flip \
#   --kv_final --k 5 --model_name Qwen/QWen3-8B\
#   --out_dir out/multitask_run \
#   --is_quant \
#   --kv_final_lambda 0.9  --kv_final_subset_multiplier 1.5 --filter 130 --max_length 128


# python test_multitask.py \
#   --dataset sst2,cr \
#   --kv_final --k 5 --model_name Qwen/QWen3-8B\
#   --out_dir out/multitask_run \
#   --is_quant \
#   --kv_final_lambda 0.9  --kv_final_subset_multiplier 1.5 --filter 130 --max_length 128
