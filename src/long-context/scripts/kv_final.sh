python test_multitask.py \
  --dataset sst2,cr,poem_sentiment \
  --do_zeroshot --use_demonstrations --kv_final --k 5 --model_name Qwen/Qwen2.5-1.5B-Instruct\
  --out_dir out/multitask_run \
  --is_quant \
  --kv_final_lambda 0.7  --kv_final_subset_multiplier 1.5