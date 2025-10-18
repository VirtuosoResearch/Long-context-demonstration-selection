python -m mta.scripts.prepare_swe_data

python -m mta.scripts.run_sweagent_vllm \
    --engine transformers \
    --model Qwen/Qwen3-0.6B \
    --device cuda \
    --limit 10 \
    --max-steps 4 \
    --agent-scaffold sweagent \
    --env-backend docker \
    --dataset SWE_Bench_Verified \
    --split test