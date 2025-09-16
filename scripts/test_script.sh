# python -m swebench.harness.run_evaluation \
#     --dataset_name princeton-nlp/SWE-bench_Lite \
#     --split test \
#     --predictions_path /home/michael/project/MTL-SWE-agents/outputs/llama-3.2-3b-instruct_lite_test/oracle_lite_test__test__meta-llama__Llama-3.2-3B-Instruct__temp-0.0__top-p-1.0.jsonl \
#     --max_workers 3 \
#     --run_id test_llama_3b-instruct

# python -m swebench.harness.run_evaluation --predictions_path gold --max_workers 4 --run_id validate-gold-test
# python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench --predictions_path gold --max_workers 4 --run_id validate-gold-all

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --split test \
    --predictions_path /home/michael/project/MTL-SWE-agents/outputs/codellama-7b-instruct_lite_test/oracle_lite_test__test__princeton-nlp__SWE-Llama-7b__temp-0.0__top-p-1.0.jsonl \
    --max_workers 3 \
    --run_id test_SWE-Llama-7b