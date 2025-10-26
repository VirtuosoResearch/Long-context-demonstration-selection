#!/usr/bin/env python
"""
Command line interface for running HumanEval agent inference.

Example
-------
Run 5 HumanEval tasks against a locally hosted vLLM server::

    python -m mta.scripts.run_humaneval_agent \
        --model your-model-name \
        --base-url http://localhost:8000/v1 \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from mta.engine.humaneval_runner import HumanEvalAgentConfig, HumanEvalAgentRunner, HumanEvalDatasetConfig
from mta.engine.sweagent_vllm import VLLMConnectionConfig
from mta.utils import compute_pass_at_k


def _json_arg(value: str) -> dict[str, Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HumanEval agent evaluation using a vLLM/OpenAI-compatible endpoint.")

    # Engine selection
    parser.add_argument("--engine", choices=["openai", "transformers"], default="openai", help="Execution backend.")

    # Connection options
    parser.add_argument("--model", required=True, help="Model name registered with the backend.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="EMPTY", help="API key expected by the backend.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p nucleus sampling parameter.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens generated per step.")
    parser.add_argument("--sampling-params", type=_json_arg, default=None, help="Additional sampling parameters as JSON.")

    # Tokenizer / local model options
    parser.add_argument("--tokenizer", default=None, help="Tokenizer name/path (transformers engine only).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code when loading tokenizer/model.")
    parser.add_argument("--tokenizer-kwargs", type=_json_arg, default=None, help="Tokenizer kwargs as JSON.")
    parser.add_argument("--model-kwargs", type=_json_arg, default=None, help="Model kwargs as JSON (transformers engine only).")
    parser.add_argument("--device", default=None, help="Target device for transformers engine (e.g., cpu, cuda, auto).")

    # Agent options
    parser.add_argument("--system-prompt", default=None, help="Override the default HumanEval system prompt.")
    parser.add_argument("--agent-extra", type=_json_arg, default=None, help="Additional agent keyword arguments as JSON.")

    # Environment options
    parser.add_argument("--timeout", type=float, default=None, help="Execution timeout for each attempt (seconds).")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum attempts per problem.")
    parser.add_argument("--env-extra", type=_json_arg, default=None, help="Additional environment kwargs as JSON.")

    # Engine behaviour
    parser.add_argument("--n-parallel", type=int, default=4, help="Number of concurrent agents.")
    parser.add_argument("--max-response-length", type=int, default=4096, help="Maximum response token length.")
    parser.add_argument("--max-prompt-length", type=int, default=2048, help="Maximum prompt token length.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum interaction steps per trajectory.")
    parser.add_argument("--trajectory-timeout", type=int, default=None, help="Optional per-trajectory timeout (seconds).")
    parser.add_argument("--max-workers", type=int, default=32, help="Thread pool size used for environment interactions.")

    # Dataset options
    parser.add_argument("--dataset-path", default=None, help="Path to a HumanEval jsonl(.gz) file. If omitted, the HuggingFace dataset is used.")
    parser.add_argument("--dataset-name", default="bigcode/humanevalpack", help="HuggingFace dataset identifier to load when no dataset path is provided.")
    parser.add_argument("--dataset-split", default="test", help="Dataset split to load from the HuggingFace dataset.")
    parser.add_argument("--language", action="append", dest="languages", default=None, help="Dataset language configuration (repeatable).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of tasks processed.")
    parser.add_argument("--task-id", action="append", dest="task_ids", default=None, help="Specific HumanEval task_id to run (repeatable).")

    # Misc
    parser.add_argument("--log-level", default="INFO", help="Root logging level.")
    parser.add_argument("--no-pass-metrics", action="store_true", help="Skip pass@k metric reporting.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger("mta.humaneval_cli")

    extra_sampling = args.sampling_params or {}
    connection = VLLMConnectionConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        extra_sampling_params=extra_sampling,
    )

    tokenizer_kwargs = args.tokenizer_kwargs.copy() if args.tokenizer_kwargs else {}
    if "trust_remote_code" not in tokenizer_kwargs:
        tokenizer_kwargs["trust_remote_code"] = args.trust_remote_code

    agent_kwargs: dict[str, Any] | HumanEvalAgentConfig | None
    if args.agent_extra:
        agent_kwargs = args.agent_extra.copy()
        if args.system_prompt is not None:
            agent_kwargs["system_prompt"] = args.system_prompt
    elif args.system_prompt is not None:
        agent_kwargs = HumanEvalAgentConfig(system_prompt=args.system_prompt)
    else:
        agent_kwargs = None

    env_kwargs = args.env_extra.copy() if args.env_extra else {}
    if args.timeout is not None:
        env_kwargs["timeout"] = args.timeout
    if args.max_attempts is not None:
        env_kwargs["max_attempts"] = args.max_attempts

    engine_kwargs: dict[str, Any] = {"max_workers": args.max_workers}
    if args.trajectory_timeout is not None:
        engine_kwargs["trajectory_timeout"] = args.trajectory_timeout

    model_kwargs = args.model_kwargs or {}

    runner = HumanEvalAgentRunner(
        connection=connection,
        engine_name=args.engine,
        tokenizer_name=args.tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        n_parallel_agents=args.n_parallel,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        max_steps=args.max_steps,
        agent_config=agent_kwargs,
        env_kwargs=env_kwargs if env_kwargs else None,
        engine_kwargs=engine_kwargs,
        device=args.device,
        model_kwargs=model_kwargs,
    )

    dataset_cfg = HumanEvalDatasetConfig()
    if args.dataset_path:
        dataset_cfg.path = args.dataset_path
    if args.limit is not None:
        dataset_cfg.limit = args.limit
    if args.task_ids:
        dataset_cfg.task_ids = args.task_ids
    if not args.dataset_path:
        dataset_cfg.dataset_name = args.dataset_name
        dataset_cfg.split = args.dataset_split
        if args.languages:
            if len(args.languages) == 1:
                dataset_cfg.language = args.languages[0]
                dataset_cfg.languages = None
            else:
                dataset_cfg.languages = args.languages
                dataset_cfg.language = None

    try:
        trajectories = runner.run_dataset(dataset_cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; exiting.")
        raise SystemExit(130) from None

    logger.info("Completed %s trajectories.", len(trajectories))

    if not args.no_pass_metrics:
        compute_pass_at_k(trajectories)


if __name__ == "__main__":
    main()
