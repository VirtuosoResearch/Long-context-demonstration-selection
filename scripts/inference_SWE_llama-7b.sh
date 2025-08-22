# python -m swebench.inference.run_llama \
#   --dataset_path princeton-nlp/SWE-bench_oracle \
#   --model_name_or_path princeton-nlp/SWE-Llama-7b \
#   --output_dir ./outputs/swe-llama-7b \
#   --split test \
#   --temperature 0 \
#   --top_p 1

python -m swebench.inference.run_llama \
  --dataset_path ./datasets/oracle_lite_test \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --output_dir ./outputs/llama-3.2-1b_lite_test \
  --split test \
  --temperature 0 \
  --top_p 1

python -m swebench.inference.run_llama \
  --dataset_path ./datasets/oracle_lite_test \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --output_dir ./outputs/llama-3.2-1b-instruct_lite_test \
  --split test \
  --temperature 0 \
  --top_p 1