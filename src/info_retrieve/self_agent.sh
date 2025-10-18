# python src/personalized/self_agent_multi.py --lang js --model Qwen/Qwen2.5-Coder-7B-Instruct
# python src/personalized/self_agent_multi.py --lang js --model codellama/CodeLlama-7b-Instruct-hf
# python src/personalized/self_agent_multi.py --lang js --model google/codegemma-7b

# python src/personalized/self_agent_multi.py --lang go --model Qwen/Qwen2.5-Coder-7B-Instruct
# python src/personalized/self_agent_multi.py --lang go --model codellama/CodeLlama-7b-Instruct-hf
# python src/personalized/self_agent_multi.py --lang go --model google/codegemma-7b

# python src/personalized/self_agent_multi.py --lang rust --model Qwen/Qwen2.5-Coder-7B-Instruct
# python src/personalized/self_agent_multi.py --lang rust --model codellama/CodeLlama-7b-Instruct-hf
# python src/personalized/self_agent_multi.py --lang rust --model google/codegemma-7b

python src/personalized/lora_agent_python.py \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --adapter_state ./out/codellama-qlora/adapter_state_dict.pt \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
#   --adapter_dir ./out/codellama-qlora \
#   --use_4bit true --bf16 true