

python src/personalized/self_agent_java.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --engine transformers \
  --max-new-tokens 256 \
  --num-iter 10 \
  --timeout 5


python src/personalized/self_agent_java.py \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --engine transformers \
  --max-new-tokens 256 \
  --num-iter 10 \
  --timeout 5


python src/personalized/self_agent_java.py \
  --model google/codegemma-7b \
  --engine transformers \
  --max-new-tokens 256 \
  --num-iter 10 \
  --timeout 5

python src/personalized/self_agent_multi.py --lang js --model Qwen/Qwen2.5-Coder-7B-Instruct
python src/personalized/self_agent_multi.py --lang js --model codellama/CodeLlama-7b-Instruct-hf
python src/personalized/self_agent_multi.py --lang js --model google/codegemma-7b

python src/personalized/self_agent_multi.py --lang go --model Qwen/Qwen2.5-Coder-7B-Instruct
python src/personalized/self_agent_multi.py --lang go --model codellama/CodeLlama-7b-Instruct-hf
python src/personalized/self_agent_multi.py --lang go --model google/codegemma-7b

python src/personalized/self_agent_multi.py --lang rust --model Qwen/Qwen2.5-Coder-7B-Instruct
python src/personalized/self_agent_multi.py --lang rust --model codellama/CodeLlama-7b-Instruct-hf
python src/personalized/self_agent_multi.py --lang rust --model google/codegemma-7b
