# python -m swebench.harness.run_evaluation \
#     --dataset_name princeton-nlp/SWE-bench_Lite \
#     --split test \
#     --predictions_path /home/michael/project/MTL-SWE-agents/outputs/llama-3.2-3b-instruct_lite_test/oracle_lite_test__test__meta-llama__Llama-3.2-3B-Instruct__temp-0.0__top-p-1.0.jsonl \
#     --max_workers 3 \
#     --run_id test_llama_3b-instruct

# python -m swebench.harness.run_evaluation \
#     --dataset_name princeton-nlp/SWE-bench_Lite \
#     --split test \
#     --predictions_path /home/michael/project/MTL-SWE-agents/outputs/swellama-7b_lite_test_ft_on_bm25_bugfix/oracle_lite_test_bugfix__test__SWE-Llama-7bbugfix_lora_merged__temp-0.0__top-p-1.0.jsonl \
#     --max_workers 3 \
#     --run_id test_lite_ftonbm25

export SWE_BENCH_NO_BUILD=1 
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --split test \
    --predictions_path ./outputs/swe-agent-lm-7b/oracle_lite_test__test__SWE-bench__SWE-agent-LM-7B__temp-0.0__top-p-1.0.jsonl \
    --max_workers 3 \
    --run_id test_swe_agent_lm_7b
