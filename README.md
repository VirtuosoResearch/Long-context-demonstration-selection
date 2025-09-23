## Multitask learning for automated software agents

Project document [https://docs.google.com/document/d/1poyVOmL3mc69hc4HIanicUG3-vxnDa8V_rsbH4LNIN8/edit?usp=sharing](https://docs.google.com/document/d/1poyVOmL3mc69hc4HIanicUG3-vxnDa8V_rsbH4LNIN8/edit?usp=sharing).

## Run open-sourced model

1. Generate patch:
```bash
python -m swebench.inference.run_llama \
  --dataset_path ./datasets/oracle_lite_test \ # This can be online datasets and local datasets.
  --split test \
  --model_name_or_path SWE-bench/SWE-agent-LM-7B \ # This can be online models and local models.
  --output_dir ./outputs/swe-agent-lm-7b \
  --temperature 0 \
  --top_p 1
```

2. Test patch:
```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \ # This can be online datasets and local datasets.
    --split test \
    --predictions_path ./outputs/swe-agent-lm-7b/oracle_lite_test__test__SWE-bench__SWE-agent-LM-7B__temp-0.0__top-p-1.0.jsonl \ # This is the path to the generated patch.
    --max_workers 3 \
    --run_id test_swe_agent_lm_7b
```
## Finetune an open-sourced model

```bash
python ./swebench/finetune_parallal.py \
    --dataset princeton-nlp/SWE-bench_bm25_13K \
    --split test \
    --model google/codegemma-7b \
    --output ./models
```

## Run agent

```bash
sweagent run-batch \
  --config config/hf_llama_model.yaml \
  --agent.model.name huggingface/meta-llama/Llama-3.2-1B-Instruct:novita \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split dev \
  --instances.slice : \
  --instances.shuffle=True
```