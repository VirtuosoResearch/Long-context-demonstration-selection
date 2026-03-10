python test_multitask.py \
  --dataset poem_sentiment,sst2,cr \
  --do_zeroshot --use_demonstrations --topk --k 3 --model_name meta-llama/Llama-3.2-1B-Instruct\
  --out_dir out/multitask_run