"""Command-line entrypoint to run the WebArena agent."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from mta.engine.sweagent_vllm import VLLMConnectionConfig
from mta.engine.webarena_runner import WebArenaRunner


def _json_arg(value: str) -> dict[str, Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the WebArena agent on BrowserGym environments.")

    parser.add_argument("--engine", choices=["openai", "transformers"], default="openai", help="Backend to use for generation.")
    parser.add_argument("--model", required=True, help="Model name or identifier.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer name/path. Defaults to the model.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible base URL (openai engine only).")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the OpenAI-compatible endpoint.")

    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p nucleus sampling value.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens generated per step.")
    parser.add_argument("--sampling-params", type=_json_arg, default=None, help="Additional sampling parameters as JSON.")

    parser.add_argument("--tokenizer-kwargs", type=_json_arg, default=None, help="Additional kwargs for AutoTokenizer.from_pretrained.")
    parser.add_argument("--model-kwargs", type=_json_arg, default=None, help="Additional kwargs for AutoModelForCausalLM.from_pretrained (transformers engine only).")
    parser.add_argument("--device", default=None, help="Device for transformers engine (e.g., cpu, cuda, auto).")

    parser.add_argument("--env-id", default="browsergym_async/webarena/airport-v1", help="BrowserGym environment identifier.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run when no task file is provided.")
    parser.add_argument("--env-kwargs", type=_json_arg, default=None, help="Additional environment kwargs as JSON.")
    parser.add_argument("--task-file", default=None, help="Path to a JSON file containing a list of tasks (each with env_id, optional task, etc.).")

    parser.add_argument("--log-level", default="INFO", help="Root logging level.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger("mta.webarena_cli")

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

    runner = WebArenaRunner(
        connection=connection,
        engine_name=args.engine,
        tokenizer_name=args.tokenizer,
        tokenizer_kwargs=args.tokenizer_kwargs,
        env_defaults=args.env_kwargs,
        model_kwargs=args.model_kwargs,
        device=args.device,
    )

    if args.task_file:
        with open(args.task_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
            if not isinstance(tasks, list):
                raise ValueError("Task file must contain a JSON list of task objects.")
    else:
        base_task = {"env_id": args.env_id}
        tasks = [base_task.copy() for _ in range(args.episodes)]

    logger.info("Running %d WebArena episodes", len(tasks))
    trajectories = runner.run(tasks)

    for idx, trajectory in enumerate(trajectories):
        reward = getattr(trajectory, "reward", None)
        logger.info("Episode %d reward: %s", idx, reward)


if __name__ == "__main__":
    main()
