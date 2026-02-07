#!/usr/bin/env python
"""
Run the POMDP MST agent on a JSON dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pomdp_mst_agent import POMDPMSTAgent, run_episode

logger = logging.getLogger(__name__)


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of entries.")
    return data


def build_llm(model_name: str, max_new_tokens: int) -> Callable[[str], str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def _llm(prompt: str) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "Return one action: query_edge(i) or select_edge(i)."},
                {"role": "user", "content": prompt},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            attention_mask = (input_ids != tokenizer.pad_token_id).to(input_ids.device)
        else:
            tokenized = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask

        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = output_ids[0][input_ids.shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _llm


def run_dataset(
    dataset: list[dict],
    llm: Callable[[str], str] | None,
    max_steps: int,
    limit: int | None,
    trace: list[dict] | None,
    print_io: bool,
) -> list[dict]:
    agent = POMDPMSTAgent(llm=llm, trace=trace, print_io=print_io)
    print("Successfully built agent")
    if limit is not None:
        dataset = dataset[:limit]
    results = []
    print("Running dataset")
    for entry in dataset:
        results.append(run_episode(entry, agent, max_steps=max_steps))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run POMDP MST agent on a JSON dataset.")
    parser.add_argument(
        "--dataset-path",
        default=os.path.join("..", "notebooks", "pomdp_mst_synthetic.json"),
        help="Path to the JSON dataset.",
    )
    parser.add_argument(
        "--output-path",
        default="pomdp_mst_agent_results.json",
        help="Path to write per-episode results.",
    )
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps per episode.")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B",
        help="HuggingFace model name or local path.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens per step.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N entries.")
    parser.add_argument(
        "--trace-path",
        default=None,
        help="Write per-step LLM inputs/outputs to this JSONL file.",
    )
    parser.add_argument(
        "--print-io",
        action="store_true",
        help="Print LLM prompt/output to stdout each step.",
    )
    parser.add_argument("--log-level", default="INFO", help="Root logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    dataset_path = os.path.abspath(args.dataset_path)
    output_path = os.path.abspath(args.output_path)

    dataset = load_dataset(dataset_path)
    llm = build_llm(args.model_name, args.max_new_tokens)
    print("Successfully built LLM: ", args.model_name)
    trace: list[dict] | None = [] if args.trace_path else None
    results = run_dataset(
        dataset,
        llm=llm,
        max_steps=args.max_steps,
        limit=args.limit,
        trace=trace,
        print_io=args.print_io,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    if args.trace_path and trace is not None:
        trace_path = os.path.abspath(args.trace_path)
        with open(trace_path, "w", encoding="utf-8") as f:
            for item in trace:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
        logger.info("Trace written to %s", trace_path)

    avg_reward = sum(item["reward"] for item in results) / max(len(results), 1)
    logger.info("Ran %s episodes", len(results))
    logger.info("Average reward: %.4f", avg_reward)
    logger.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
