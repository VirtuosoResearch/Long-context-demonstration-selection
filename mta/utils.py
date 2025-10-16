"""
Lightweight utility functions replicated from rllm for the mta namespace.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Iterable

from mta.agents import Trajectory


def compute_pass_at_k(results: Iterable[Trajectory]) -> dict[str, float]:
    """
    Compute pass@1 and pass@k for a collection of trajectories.

    This mirrors the implementation in ``rllm.utils`` but avoids importing
    optional heavy dependencies.
    """
    problem_correct_map: defaultdict[str, int] = defaultdict(int)
    problem_total_map: defaultdict[str, int] = defaultdict(int)

    for trajectory in results:
        task = getattr(trajectory, "task", None)
        if isinstance(task, dict):
            problem_str = json.dumps(task, sort_keys=True)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        is_correct = 1 if getattr(trajectory, "reward", 0) > 0 else 0
        problem_correct_map[problem_hash] += is_correct
        problem_total_map[problem_hash] += 1

    total_problems = len(problem_correct_map)
    total_attempts = sum(problem_total_map.values())

    pass_at_1 = sum(problem_correct_map.values()) / total_attempts if total_attempts else 0.0
    pass_at_k = (
        sum(1 for correct in problem_correct_map.values() if correct > 0) / total_problems if total_problems else 0.0
    )

    metrics = {
        "total_problems": float(total_problems),
        "total_attempts": float(total_attempts),
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
    }

    print("Total unique problems:", int(total_problems))
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@k Accuracy:", pass_at_k)

    return metrics
