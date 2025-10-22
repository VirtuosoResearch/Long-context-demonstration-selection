python src/info_retrieve/sft_agent.py \
  --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
  --humaneval_language python \
  --output_dir ./out/qwen-coder-qlora \
  --batch_size 4 --grad_accum 2 --epochs 10 --lr 5e-5
# python src/personalized/sft_agent.py \
#   --model_name codellama/CodeLlama-7b-Instruct-hf \
#   --humaneval_language python \
#   --output_dir ./out/codellama-qlora \
#   --batch_size 2 --grad_accum 4 --epochs 30 --lr 2e-4