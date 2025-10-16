python src/personalized/self_agent.py \
  --model google/codegemma-7b \
  --engine transformers \
  --max-new-tokens 256 \
  --num-iter 10 \
  --timeout 5


python src/personalized/self_agent_cpp.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --engine transformers \
  --max-new-tokens 256 \
  --num-iter 10 \
  --timeout 5


python src/personalized/self_agent_cpp.py \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --engine transformers \
  --max-new-tokens 256 \
  --num-iter 10 \
  --timeout 5


python src/personalized/self_agent_cpp.py \
  --model google/codegemma-7b \
  --engine transformers \
  --max-new-tokens 256 \
  --num-iter 10 \
  --timeout 5


