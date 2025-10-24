#!/usr/bin/env python
"""
Quick diagnostic script for checking whether the configured OpenAI-compatible
completion endpoint is reachable before launching a full SWE-agent run.

Example:
    python -m mta.scripts.debug_llm_endpoint \
        --model agentica-org/DeepSWE-Preview \
        --base-url https://your-api-host/v1 \
        --api-key sk-... \
        --prompt "Hello from debug probe" \
        --max-tokens 64
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from openai import AsyncOpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ping an OpenAI-compatible completions endpoint.")
    parser.add_argument("--model", required=True, help="Model identifier expected by the endpoint.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the endpoint.")
    parser.add_argument("--prompt", default="ping", help="Prompt to send to the completions endpoint.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Maximum tokens to sample in the response.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling parameter.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


async def probe_endpoint(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("mta.debug_llm_endpoint")

    client = AsyncOpenAI(base_url=args.base_url.rstrip("/"), api_key=args.api_key)
    logger.info("Sending probe to %s (model=%s)", args.base_url, args.model)

    sampling_kwargs = {"max_tokens": args.max_tokens, "temperature": args.temperature}
    if args.top_p is not None:
        sampling_kwargs["top_p"] = args.top_p

    response = await client.completions.create(model=args.model, prompt=args.prompt, **sampling_kwargs)

    text = response.choices[0].text if response and response.choices else ""
    logger.info("Raw response object: %s", response)
    logger.info("Generated text: %s", text.strip())


def main() -> None:
    args = parse_args()
    asyncio.run(probe_endpoint(args))


if __name__ == "__main__":
    main()
