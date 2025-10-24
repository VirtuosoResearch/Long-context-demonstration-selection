# mta – Multi-task agent utilities for vLLM-backed SWE inference

This package mirrors the relevant structure of `rllm` while trimming it down to
the pieces needed for SWE-agent style inference with a custom vLLM deployment.
It re-exports the core data structures (`Step`, `Trajectory`, `AgentExecutionEngine`)
so existing code can reuse the same abstractions without depending on the full
`rllm` namespace.

## Quick start

0. Install the mta package into your environment (once per virtualenv)::

   ```bash
   pip install -e .
   ```

   To include the optional R2E-Gym environment dependencies run::

   ```bash
   pip install -e .[env]
   ```

1. Launch a vLLM server that exposes an OpenAI-compatible completion endpoint::

   ```bash
   export MODEL=agentica-org/DeepSWE-Preview
   VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve "$MODEL" \
       --host 0.0.0.0 \
       --port 8000 \
       --max-model-len 65536 \
       --tensor-parallel-size 1
   ```

2. Prepare the SWE-Bench dataset using the bundled helper (once per machine)::

   ```bash
   python -m mta.scripts.prepare_swe_data
   ```

3. Run inference via the CLI::

   ```bash
   python -m mta.scripts.run_sweagent_vllm \
       --engine openai \
       --model "$MODEL" \
       --base-url http://localhost:8000/v1 \
       --dataset SWE_Bench_Verified \
       --split test \
       --limit 5
   ```

The script loads trajectories via `DatasetRegistry`, spins up the shared
`AgentExecutionEngine`, and computes pass@k metrics on completion.

To run the same workflow entirely locally with Hugging Face weights::

   ```bash
   python -m mta.scripts.run_sweagent_vllm \
       --engine transformers \
       --model Qwen/Qwen3-0.6B \
       --device cuda \
       --dataset SWE_Bench_Verified \
       --split test \
       --limit 5
   ```

## Customisation tips

- Change `--agent-scaffold` to `r2egym` for prompts matching the rllm setup.
- Use `--engine transformers` (optionally with `--device cuda` and `--model-kwargs='{"torch_dtype": "bfloat16"}'`) to load Hugging Face models directly without an HTTP server.
- Use `--sampling-params='{"temperature":0.8,"top_p":0.9}'` to forward advanced
  sampling arguments directly to vLLM.
- Provide `--env-extra` with JSON to pass additional `SWEEnv` keyword arguments,
  e.g. `--env-extra='{"verbose": true}'`.

Refer to `mta/scripts/run_sweagent_vllm.py --help` for the full list of options.

## HumanEval agent evaluation

The same execution engine now supports HumanEval-style problems through a lightweight agent scaffold.

Make sure the bundled `human-eval` package (or any equivalent installation providing `human_eval.execution`) is available on your `PYTHONPATH`, e.g.:

```bash
pip install -e human-eval
```

1. (Optional) materialise the dataset into the local registry to inspect it later::

   ```bash
   python -m mta.scripts.prepare_humaneval_data
   ```

2. Launch evaluation via the bundled CLI (OpenAI/vLLM endpoint) ::

   ```bash
   python -m mta.scripts.run_humaneval_agent \
       --model your-model-name \
       --base-url http://localhost:8000/v1 \
       --limit 10
   ```

   To run entirely locally with Hugging Face weights, swap `--engine transformers` and provide `--device` / `--model-kwargs` as needed.

Pass `--help` to either script for the exhaustive set of flags (custom prompts, attempt limits, dataset path overrides, etc.).
