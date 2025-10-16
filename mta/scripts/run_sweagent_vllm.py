#!/usr/bin/env python
"""
Command line interface for running SWE-agent inference backed by vLLM.

Example
-------
Run 10 SWE-Bench Verified tasks against a locally hosted vLLM server::

    python -m mta.scripts.run_sweagent_vllm \
        --model agentica-org/DeepSWE-Preview \
        --base-url http://localhost:8000/v1 \
        --dataset SWE_Bench_Verified \
        --split test \
        --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from mta.engine.sweagent_vllm import SweAgentConfig, SweAgentVLLMRunner, VLLMConnectionConfig
from mta.utils import compute_pass_at_k


def _json_arg(value: str) -> dict[str, Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWE-agent inference using a vLLM OpenAI server.")

    # Engine selection
    parser.add_argument(
        "--engine",
        choices=["openai", "transformers"],
        default="openai",
        help="Execution backend: 'openai' for HTTP endpoints, 'transformers' for local generation.",
    )

    # vLLM / connection options
    parser.add_argument("--model", required=True, help="Model name registered with the vLLM server.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible base URL exposed by vLLM.")
    parser.add_argument("--api-key", default="EMPTY", help="API key expected by the vLLM server.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling parameter.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens generated per step.")
    parser.add_argument(
        "--sampling-params",
        type=_json_arg,
        default=None,
        help="Additional sampling parameters as JSON (merged into the vLLM request).",
    )

    # Tokenizer
    parser.add_argument("--tokenizer", default=None, help="Tokenizer name/path. Defaults to the model argument.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code when loading the tokenizer.")
    parser.add_argument(
        "--tokenizer-kwargs",
        type=_json_arg,
        default=None,
        help="Extra kwargs for AutoTokenizer.from_pretrained as JSON.",
    )

    parser.add_argument("--model-kwargs", type=_json_arg, default=None, help="Extra kwargs for AutoModelForCausalLM as JSON (transformers engine only).")
    parser.add_argument("--device", default=None, help="Target device for local transformers engine (e.g., cpu, cuda, auto).")

    # Agent / environment behaviour
    parser.add_argument("--n-parallel", type=int, default=8, help="Number of concurrent agents.")
    parser.add_argument("--max-response-length", type=int, default=65536, help="Maximum response token length.")
    parser.add_argument("--max-prompt-length", type=int, default=4096, help="Maximum prompt length forwarded to vLLM.")
    parser.add_argument("--max-steps", type=int, default=40, help="Max interaction steps per trajectory.")
    parser.add_argument("--use-fn-calling", action="store_true", help="Toggle SWEAgent function-calling parsing.")
    parser.add_argument(
        "--format-model-response",
        action="store_true",
        help="Format the model response stored in the trajectory as thought + action XML.",
    )
    parser.add_argument("--agent-scaffold", default="sweagent", help="Prompt scaffold passed to SWEAgent.")

    parser.add_argument("--env-backend", default="docker", help="Backend used by SWEEnv (docker, kubernetes, ...).")
    parser.add_argument("--env-step-timeout", type=int, default=90, help="Step timeout forwarded to SWEEnv.")
    parser.add_argument("--env-reward-timeout", type=int, default=300, help="Reward timeout forwarded to SWEEnv.")
    parser.add_argument(
        "--env-extra",
        type=_json_arg,
        default=None,
        help="Additional SWEEnv keyword arguments as JSON.",
    )

    # Dataset options
    parser.add_argument("--dataset", default="SWE_Bench_Verified", help="Dataset name registered with DatasetRegistry.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of tasks processed.")

    # Engine behaviour
    parser.add_argument("--trajectory-timeout", type=int, default=None, help="Optional per-trajectory timeout (seconds).")
    parser.add_argument("--max-workers", type=int, default=64, help="Thread pool size used for environment interactions.")

    # Misc
    parser.add_argument("--log-level", default="INFO", help="Root logging level.")
    parser.add_argument("--no-pass-metrics", action="store_true", help="Skip pass@k metric reporting.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger("mta.sweagent_vllm_cli")

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

    tokenizer_kwargs = args.tokenizer_kwargs or {}
    if "trust_remote_code" not in tokenizer_kwargs:
        tokenizer_kwargs["trust_remote_code"] = args.trust_remote_code

    agent_cfg = SweAgentConfig(
        use_fn_calling=args.use_fn_calling,
        format_model_response=args.format_model_response,
        scaffold=args.agent_scaffold,
    )

    env_kwargs = {
        "backend": args.env_backend,
        "step_timeout": args.env_step_timeout,
        "reward_timeout": args.env_reward_timeout,
        "scaffold": args.agent_scaffold,  # Align env prompts with agent prompts by default.
    }
    if args.env_extra:
        env_kwargs.update(args.env_extra)

    engine_kwargs: dict[str, Any] = {"max_workers": args.max_workers}
    if args.trajectory_timeout is not None:
        engine_kwargs["trajectory_timeout"] = args.trajectory_timeout

    model_kwargs = args.model_kwargs or {}

    runner = SweAgentVLLMRunner(
        connection=connection,
        engine_name=args.engine,
        tokenizer_name=args.tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        n_parallel_agents=args.n_parallel,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        max_steps=args.max_steps,
        agent_config=agent_cfg,
        env_kwargs=env_kwargs,
        engine_kwargs=engine_kwargs,
        device=args.device,
        model_kwargs=model_kwargs,
    )

    try:
        trajectories = runner.run_dataset(args.dataset, args.split, limit=args.limit)
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
