python test_multitask.py \
  --dataset sst2,cr,coin_flip,addition,poem_sentiment \
  --gradicl --k 5 \
  --filter 30 \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --out_dir out/multitask_run \
  --is_quant --max_length 70