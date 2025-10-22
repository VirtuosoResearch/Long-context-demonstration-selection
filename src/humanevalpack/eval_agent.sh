python src/humanevalpack/eval_agent_multi.py \
    --lang python \
    --model google/codegemma-7b \
    --num_iter 10 \

# python src/info_retrieve/lora_agent_python2.py \
#   --model codellama/CodeLlama-7b-Instruct-hf \
#   --adapter_state ./src/info_retrieve/out/codellama-qlora-python_100/adapter_state_dict.pt \
#   --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
#   --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
# #   --adapter_dir ./out/codellama-qlora \
# #   --use_4bit true --bf16 true