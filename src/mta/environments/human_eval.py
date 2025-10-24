from __future__ import annotations

import copy
import re
from typing import Any, Dict

from human_eval.execution import check_correctness

from mta.environments.base import BaseEnv


class HumanEvalEnv(BaseEnv):
    """Environment wrapper that evaluates HumanEval completions via unit tests."""

    DEFAULT_TIMEOUT = 3.0
    DEFAULT_MAX_ATTEMPTS = 3

    def __init__(
        self,
        *,
        problem: Dict[str, Any],
        timeout: float | None = None,
        max_attempts: int | None = None,
    ):
        if problem is None:
            raise ValueError("A HumanEval problem dictionary must be provided.")

        self.problem = copy.deepcopy(problem)
        self.timeout = float(self.problem.get("timeout", timeout or self.DEFAULT_TIMEOUT))
        self.max_attempts = int(self.problem.get("max_attempts", max_attempts or self.DEFAULT_MAX_ATTEMPTS))

        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer.")

        self._attempts = 0
        self._done = False
        self._last_reward = 0.0

    # ------------------------------------------------------------------ #
    # BaseEnv implementation
    # ------------------------------------------------------------------ #

    def reset(self) -> tuple[str, dict]:
        self._attempts = 0
        self._done = False
        self._last_reward = 0.0

        instructions = self.problem.get("instructions")
        if not instructions:
            instructions = self._build_default_instructions()

        info = {
            "task_id": self.problem.get("task_id"),
            "entry_point": self.problem.get("entry_point"),
            "max_attempts": self.max_attempts,
        }

        return instructions, info

    def step(self, action: Any) -> tuple[str, float, bool, dict]:
        if self._done:
            info = {
                "task_id": self.problem.get("task_id"),
                "entry_point": self.problem.get("entry_point"),
                "attempt": self._attempts,
                "max_attempts": self.max_attempts,
                "passed": self._last_reward > 0,
                "result": "completed",
            }
            return "Task already completed. No further submissions are required.", self._last_reward, True, info

        attempt_id = self._attempts
        submission = self._extract_completion(str(action) if action is not None else "")

        if not submission.strip():
            feedback = (
                "No Python implementation detected in your response. "
                "Please reply with a Python code block that completes the provided function body."
            )
            self._attempts += 1
            self._update_state(passed=False)

            info = {
                "task_id": self.problem.get("task_id"),
                "entry_point": self.problem.get("entry_point"),
                "attempt": self._attempts,
                "max_attempts": self.max_attempts,
                "passed": False,
                "result": "empty submission",
            }

            done = self._done
            return feedback, self._last_reward, done, info

        result = check_correctness(self.problem, submission, self.timeout, completion_id=attempt_id)
        passed = bool(result.get("passed", False))

        self._attempts += 1
        self._update_state(passed=passed)

        status = "PASSED" if passed else "FAILED"
        feedback_lines = [
            f"Evaluation result ({status}) on attempt {self._attempts}: {result.get('result', 'unknown')}.",
        ]
        if not passed and not self._done:
            feedback_lines.append("Please submit a revised Python code block containing only the function body.")
        elif not passed and self._done:
            feedback_lines.append("Maximum attempts reached. No further submissions will be evaluated.")
        else:
            feedback_lines.append("All unit tests passed. You may stop responding.")

        info = {
            "task_id": self.problem.get("task_id"),
            "entry_point": self.problem.get("entry_point"),
            "attempt": self._attempts,
            "max_attempts": self.max_attempts,
            "passed": passed,
            "result": result.get("result"),
        }

        return "\n".join(feedback_lines), self._last_reward, self._done, info

    def compute_final_reward(self):
        return self._last_reward

    def close(self):
        return

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_dict(extra_info: dict | str) -> "HumanEvalEnv":
        if isinstance(extra_info, str):
            import json

            problem_dict = json.loads(extra_info)
        else:
            problem_dict = extra_info

        problem = copy.deepcopy(problem_dict)
        timeout = problem.pop("timeout", None)
        max_attempts = problem.pop("max_attempts", None)

        return HumanEvalEnv(problem=problem, timeout=timeout, max_attempts=max_attempts)

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_default_instructions(self) -> str:
        task_id = self.problem.get("task_id", "unknown")
        entry_point = self.problem.get("entry_point", "the target function")
        prompt = self.problem.get("prompt", "").rstrip()

        header = [
            f"Task: {task_id}",
            "Implement the body of the provided Python function.",
            f"Return your answer as a Python code block that completes `{entry_point}`.",
            f"You may attempt up to {self.max_attempts} submission(s); feedback will be provided after each attempt.",
        ]

        return "\n\n".join(header + ["---", prompt])

    def _update_state(self, *, passed: bool):
        self._last_reward = 1.0 if passed else 0.0
        if passed:
            self._done = True
        elif self._attempts >= self.max_attempts:
            self._done = True
        else:
            self._done = False

    def _extract_completion(self, action_text: str) -> str:
        code = self._extract_code_block(action_text)
        if not code:
            code = action_text.strip()

        if not code:
            return ""

        code = self._normalize_definition_wrapper(code)
        if not code.endswith("\n"):
            code += "\n"

        return code

    def _extract_code_block(self, text: str) -> str:
        pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip("\n")
        return ""

    def _normalize_definition_wrapper(self, code: str) -> str:
        entry_point = self.problem.get("entry_point")
        if not entry_point:
            return code

        definition_pattern = re.compile(rf"def\s+{re.escape(entry_point)}\s*\(.*?\):\s*(?:#.*)?", re.DOTALL)
        match = definition_pattern.search(code)
        if not match:
            return self._format_body(code)

        body = code[match.end() :].lstrip("\n")
        if not body:
            return ""

        return self._format_body(body)

    def _format_body(self, body: str) -> str:
        trimmed = body.strip("\n")
        if not trimmed:
            return ""

        lines = trimmed.splitlines()
        indent_candidates = [
            len(line) - len(line.lstrip())
            for line in lines
            if line.strip() and (len(line) - len(line.lstrip())) > 0
        ]
        base_indent = min(indent_candidates) if indent_candidates else 0

        normalized_lines: list[str] = []
        for line in lines:
            if line.strip():
                stripped = line.lstrip()
                indent_len = len(line) - len(stripped)
                relative_indent = max(0, indent_len - base_indent)
                normalized_lines.append(" " * (4 + relative_indent) + stripped)
            else:
                normalized_lines.append("")

        normalized = "\n".join(normalized_lines).rstrip("\n")
        return normalized
