python streaming_llm_baseline.py \
      --dataset sst2 \
      --model_name Qwen/Qwen2.5-1.5B-Instruct \
      --k 50 --sink_tokens 4 \
      --recent_tokens_list 16,32,64,128,256 \
      --eval_split dev --retrieval_split test