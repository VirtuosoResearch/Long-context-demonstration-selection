python kv_cache_inference.py \
  --dataset poem_sentiment \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 50 \
  --demo_strategy first \
  --device cuda --run_mode both

python kv_cache_inference.py \
  --dataset poem_sentiment \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 100 \
  --demo_strategy first \
  --device cuda --run_mode both

python kv_cache_inference.py \
  --dataset poem_sentiment \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 150 \
  --demo_strategy first \
  --device cuda --run_mode both



python kv_cache_inference.py \
  --dataset sst2 \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 50 \
  --demo_strategy first \
  --device cuda --run_mode both

python kv_cache_inference.py \
  --dataset sst2 \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 100 \
  --demo_strategy first \
  --device cuda --run_mode both

python kv_cache_inference.py \
  --dataset sst2 \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 150 \
  --demo_strategy first \
  --device cuda --run_mode both



python kv_cache_inference.py \
  --dataset cr \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 50 \
  --demo_strategy first \
  --device cuda --run_mode both

python kv_cache_inference.py \
  --dataset cr \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 100 \
  --demo_strategy first \
  --device cuda --run_mode both

python kv_cache_inference.py \
  --dataset cr \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 150 \
  --demo_strategy first \
  --device cuda --run_mode both




python kv_cache_inference.py \
  --dataset coin_flip \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 50 \
  --demo_strategy first \
  --device cuda --run_mode both

python kv_cache_inference.py \
  --dataset coin_flip \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 100 \
  --demo_strategy first \
  --device cuda --run_mode both

python kv_cache_inference.py \
  --dataset coin_flip \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --k 150 \
  --demo_strategy first \
  --device cuda --run_mode both