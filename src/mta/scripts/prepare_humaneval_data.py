#!/usr/bin/env python
"""
Utility to materialise the HumanEval dataset into the mta DatasetRegistry.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from human_eval.data import HUMAN_EVAL, read_problems

from mta.data import DatasetRegistry

logger = logging.getLogger(__name__)


def prepare_humaneval_data(
    *,
    dataset_path: str = HUMAN_EVAL,
    timeout: float = 3.0,
    max_attempts: int = 3,
) -> Any:
    problems = read_problems(dataset_path)
    records = []

    for task_id in sorted(problems.keys()):
        problem = problems[task_id]
        records.append(
            {
                "task_id": problem["task_id"],
                "prompt": problem["prompt"],
                "entry_point": problem["entry_point"],
                "test": problem["test"],
                "timeout": timeout,
                "max_attempts": max_attempts,
            }
        )

    logger.info("Registering %s HumanEval problem(s) with timeout=%s max_attempts=%s", len(records), timeout, max_attempts)
    dataset = DatasetRegistry.register_dataset("HumanEval", records, "test")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register the HumanEval dataset with the mta DatasetRegistry.")
    parser.add_argument("--dataset-path", default=HUMAN_EVAL, help="Path to the HumanEval jsonl(.gz) file.")
    parser.add_argument("--timeout", type=float, default=3.0, help="Per-attempt execution timeout (seconds).")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum attempts allowed per task.")
    parser.add_argument("--log-level", default="INFO", help="Root logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    dataset = prepare_humaneval_data(dataset_path=args.dataset_path, timeout=args.timeout, max_attempts=args.max_attempts)
    sample = dataset.get_data()[0] if dataset.get_data() else None

    print("Registered dataset:", dataset.name)
    print("Split:", dataset.split)
    print("Total problems:", len(dataset))
    if sample:
        print("Sample entry:", sample["task_id"])


if __name__ == "__main__":
    main()
