python -m mta.scripts.prepare_swe_data

python -m mta.scripts.run_sweagent_vllm \
    --engine transformers \
    --model Qwen/Qwen3-0.6B \
    --device cuda \
    --limit 1 \
    --max-steps 4 \
    --agent-scaffold sweagent \
    --env-backend docker \
    --dataset SWE_Bench_Verified \
    --split test

python -m mta.scripts.run_sweagent_vllm \
    --engine openai \
    --model gpt-4o-mini \
    --base-url https://api.openai.com/v1 \
    --api-key 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 \
    --dataset SWE_Bench_Verified \
    --split test \
    --limit 1 \
    --n-parallel 4 \
    --max-response-length 10000 \
    --max-prompt-length 20000 \
    --max-steps 4 \
    --agent-scaffold sweagent \
    --env-backend docker \
    --use-fn-calling

# WebArena example (requires BrowserGym and appropriate credentials)
# python -m mta.scripts.run_webarena_agent \
#     --engine openai \
#     --model deepseek-chat \
#     --tokenizer deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
#     --base-url https://api.deepseek.com/v1 \
#     --api-key $DEEPSEEK_API_KEY \
#     --env-id browsergym_async/webarena/airport \
#     --episodes 1

python -m mta.scripts.run_webarena_agent \
    --engine openai \
    --model gpt-4o-mini \
    --base-url https://api.openai.com/v1 \
    --api-key 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 \
    --api-key sk-4e15ca63b2c042c8ba1f6fbc20f3c4fd \
    --env-id browsergym/webarena \
    --episodes 1

python -m mta.scripts.debug_llm_endpoint \
    --model gpt-4o-mini \
    --base-url https://api.openai.com/v1 \
    --api-key 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 \
    --prompt "SWE-agent connectivity check" \
    --max-tokens 64 \
    --temperature 0.0


PYTHONPATH=human-eval python -m mta.scripts.run_humaneval_agent \
    --engine openai \
    --model gpt-4o-mini \
    --base-url https://api.openai.com/v1 \
    --api-key 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 \
    --limit 1 \
    --n-parallel 4 \
    --max-response-length 10000 \
    --max-prompt-length 10000 \
    --max-steps 4 \
    --temperature 1 \
    --language python
    
