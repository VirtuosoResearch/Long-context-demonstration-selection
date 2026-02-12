python run_pomdp_mst_agent.py --print-io --limit 100 --model-name Qwen/Qwen2.5-3B-Instruct \
    --trajectory-success-path ../notebooks/pomdp_mst_success.json \
    --trajectory-suboptimal-path ../notebooks/pomdp_mst_suboptimal.json \
    --trajectory-success-ratio 1 \
    --trajectory-count 10