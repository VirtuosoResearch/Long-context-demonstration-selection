python test_multitask.py \
  --dataset addition,coin_flip \
  --topk --k 5 --model_name Qwen/Qwen2.5-1.5B-Instruct\
  --out_dir out/multitask_run \
  --is_quant --max_length 90