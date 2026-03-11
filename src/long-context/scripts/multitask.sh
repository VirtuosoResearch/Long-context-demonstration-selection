python test_multitask.py \
  --dataset sst2,cr,poem_sentiment \
  --do_zeroshot --use_demonstrations --topk --k 100 --model_name Qwen/Qwen2.5-1.5B-Instruct\
  --out_dir out/multitask_run \
  --is_quant