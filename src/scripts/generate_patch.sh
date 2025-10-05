# python -m swebench.inference.run_llama \
#   --dataset_path princeton-nlp/SWE-bench_oracle \
#   --model_name_or_path princeton-nlp/SWE-Llama-7b \
#   --output_dir ./outputs/swe-llama-7b \
#   --split test \
#   --temperature 0 \
#   --top_p 1

# python -m swebench.inference.run_llama \
#   --dataset_path ./datasets/oracle_lite_test \
#   --model_name_or_path meta-llama/Llama-3.2-1B \
#   --output_dir ./outputs/llama-3.2-1b_lite_test \
#   --split test \
#   --temperature 0 \
#   --top_p 1

# python -m swebench.inference.run_llama \
#   --dataset_path ./datasets/oracle_lite_test \
#   --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
#   --output_dir ./outputs/llama-3.2-3b-instruct_lite_test \
#   --split test \
#   --temperature 0 \
#   --top_p 1

# codellama/CodeLlama-7b-Instruct-hf

# python -m swebench.inference.run_llama \
#   --dataset_path /home/michael/project/MTL-SWE-agents/datasets/oracle_lite_test_bugfix \
#   --model_name_or_path /home/michael/project/MTL-SWE-agents/model/SWE-Llama-7bbugfix_lora_merged \
#   --output_dir ./outputs/swellama-7b_lite_test_ft_on_bm25_bugfix \
#   --split test \
#   --temperature 0 \
#   --top_p 1

# python -m swebench.inference.run_llama \
#   --dataset_path /home/michael/project/MTL-SWE-agents/datasets/oracle_lite_test_feature \
#   --model_name_or_path /home/michael/project/MTL-SWE-agents/model/SWE-Llama-7bfeature_lora_merged \
#   --output_dir ./outputs/swellama-7b_lite_test_ft_on_bm25_feature \
#   --split test \
#   --temperature 0 \
#   --top_p 1

python -m swebench.inference.run_llama \
  --dataset_path ./datasets/oracle_lite_test \
  --model_name_or_path SWE-bench/SWE-agent-LM-7B \
  --output_dir ./outputs/swe-agent-lm-7b \
  --split test \
  --temperature 0 \
  --top_p 1

# python -m swebench.inference.run_llama \
#   --dataset_path ./datasets/oracle_lite_test \
#   --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
#   --output_dir ./outputs/codellama-7b-instruct_lite_test \
#   --split test \
#   --temperature 0 \
  # --top_p 1
