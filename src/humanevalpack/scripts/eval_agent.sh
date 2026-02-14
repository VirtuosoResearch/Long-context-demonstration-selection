CUDA_VISIBLE_DEVICES=0,1 python eval_agent_multi.py \
    --lang python \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --num_iter 10 \
    --print_attribution \
    --attribution_method integrated_gradients \
    --ig_steps 20 \
    --sentence_level \

CUDA_VISIBLE_DEVICES=0,1 python eval_agent_multi.py \
    --lang cpp \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --num_iter 10 \
    --print_attribution \
    --attribution_method integrated_gradients \
    --ig_steps 20 \
    --sentence_level \

# python src/info_retrieve/lora_agent_python2.py \
#   --model codellama/CodeLlama-7b-Instruct-hf \
#   --adapter_state ./src/info_retrieve/out/codellama-qlora-python_100/adapter_state_dict.pt \
#   --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
#   --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
# #   --adapter_dir ./out/codellama-qlora \
# #   --use_4bit true --bf16 true