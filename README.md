## Multitask learning for automated software agents

Project document [https://docs.google.com/document/d/1poyVOmL3mc69hc4HIanicUG3-vxnDa8V_rsbH4LNIN8/edit?usp=sharing](https://docs.google.com/document/d/1poyVOmL3mc69hc4HIanicUG3-vxnDa8V_rsbH4LNIN8/edit?usp=sharing).

## Environment

We provide the [environment file](./environment.yml) including the package versions we used in the experiments. For optimal reproducibility, we recommend using the same package versions.

## System structure

The structure of our system is:

```bash
./classified/ # classified mapping of the dataset, conclude in Json version

./datasets/ # cache of different datasets

./logs/run_evaluation # logs of each evaluation
└── test_gold_all # we use gold test as an example
    └── astropy__astropy-6938 # a single instance within the dataset
        └── eval.sh # scripts to run the patch
        └── patch.diff # generated patch
        └── report.json # conclusion report of this single running
        └── run_instance.log # logs of running the instance
        └── test_output.txt # detailed running output in the terminal
    ......
    └── sympy__sympy-24909

./outputs/ # location of generated patch
└── swe-llama-7b # use swe-llama-7b as an example
    └── princeton-nlp__SWE-bench_oracle__test__princeton-nlp__SWE-Llama-7b__temp-0.0__top-p-1.0.json # the generated patches are in Json version

./results/ # conclusion report of the whole running

./scripts/ # scripts to fetch offline image, generate patch, finetune, run agent, and test patch.

./SWE-agent/ # source code for agent usage

./swebench/ # source code for LLM usage

```

The soft evaluation and offline evaluation are enrolled in the source codes.

## Run open-sourced model

1. Generate patch.
```bash
python -m swebench.inference.run_llama \
  --dataset_path ./datasets/oracle_lite_test \ # This can be online datasets and local datasets.
  --split test \
  --model_name_or_path SWE-bench/SWE-agent-LM-7B \ # This can be online models and local models.
  --output_dir ./outputs/swe-agent-lm-7b \ # the patch will be generated in this path.
  --temperature 0 \
  --top_p 1
```

2. Test patch.
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

1. Use open-sourced model to evaluate.
```bash
cd ./SWE-agent
sweagent run-batch \
  --config config/hf_llama_model.yaml \
  --agent.model.name huggingface/meta-llama/Llama-3.2-1B-Instruct:novita \ # This should be the api version of the open-sourced model.
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split dev \
  --instances.slice :3 \
  --instances.shuffle=True
```

2. Use api to evaluate.
```bash
cd ./SWE-agent
sweagent run-batch \
    --config config/default.yaml \
    --agent.model.name gpt-4o \
    --agent.model.per_instance_cost_limit 2.00 \
    --instances.type swe_bench \
    --instances.subset lite \
    --instances.split dev  \
    --instances.slice :3 \
    --instances.shuffle=True
```

## Utilize offline evaluation
To accelerate the evaluation process, please run following commands to fetch offline images.
```bash
python ./notebooks/fetch_offline_image.py
```

## Evaluate HumanEvalPack
Here is the script to evaluate HumanEvalPack dataset.
```bash
bash ./src/personalized/self_agent.sh
```

If you want to finetune your own model, please run:
```bash
bash ./src/personalized/sft_agent.sh
```
LoRA and QLoRA are both supported.