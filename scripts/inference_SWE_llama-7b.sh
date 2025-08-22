python -m swebench.inference.run_llama \
  --dataset_path princeton-nlp/SWE-bench_oracle \
  --model_name_or_path princeton-nlp/SWE-Llama-7b \
  --output_dir ./outputs/swe-llama-7b \
  --split test \
  --temperature 0 \
  --top_p 1