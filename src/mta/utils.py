"""
Lightweight utility functions replicated from rllm for the mta namespace.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
import re
import threading
from typing import Callable, Iterable

from mta.agents import Trajectory
from mta.task_prompts import available_languages

LanguageNormalizer = Callable[[str], str]


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


@dataclass
class LanguageDetectionConfig:
    default_language: str = "python"
    detection_temperature: float = 0.0
    use_heuristics: bool = True
    fallback_languages: tuple[str, ...] = (
        "python",
        "javascript",
        "java",
        "cpp",
        "c",
        "go",
        "rust",
        "typescript",
    )


class LanguageDetector:
    """Detect the target programming language for a coding task."""

    def __init__(
        self,
        *,
        engine_name: str,
        connection_kwargs: dict[str, str],
        model_name: str,
        config: LanguageDetectionConfig | None = None,
        normalizer: LanguageNormalizer | None = None,
    ):
        self.engine_name = engine_name
        self.connection_kwargs = connection_kwargs
        self.model_name = model_name
        self.config = config or LanguageDetectionConfig()
        self._normalizer = normalizer or (lambda text: text.lower().strip())
        self._client = None
        self._client_lock = threading.Lock()
        self._last_source = "default"

        self._initialize_client()

    def detect(self, task_description: str, fallback_language: str | None = None) -> str:
        fallback = fallback_language or self.config.default_language
        if not task_description.strip():
            self._last_source = "default"
            return fallback

        llm_guess = self._detect_with_llm(task_description)
        if llm_guess:
            self._last_source = "llm"
            return llm_guess

        if self.config.use_heuristics:
            heuristic_guess = self._detect_with_heuristics(task_description)
            if heuristic_guess:
                self._last_source = "heuristic"
                return heuristic_guess

        self._last_source = "default"
        return fallback

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _initialize_client(self):
        if self.engine_name != "openai":
            return

        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return

        base_url = self.connection_kwargs.get("base_url")
        api_key = self.connection_kwargs.get("api_key")
        if not api_key or api_key.upper() == "EMPTY":
            return

        try:
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        except Exception:
            self._client = None

    def _detect_with_llm(self, text: str) -> str | None:
        if self._client is None:
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a classifier that identifies the programming language required by a coding task. "
                    "Reply with a single lowercase word naming the language (for example: python, javascript, java, c, cpp, go). "
                    "If the language cannot be determined, reply with 'unknown'."
                ),
            },
            {
                "role": "user",
                "content": f"Task description:\n{text}\n\nLanguage:",
            },
        ]

        with self._client_lock:
            try:
                response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                    model=self.model_name,
                    messages=messages,
                    temperature=self.config.detection_temperature,
                    max_tokens=5,
                )
            except Exception:
                return None

        try:
            candidate = response.choices[0].message.content  # type: ignore[index]
        except (AttributeError, IndexError):
            return None

        if not candidate:
            return None
        language = self._normalizer(candidate.split()[0])
        if language == "unknown":
            return None
        return language

    def _detect_with_heuristics(self, text: str) -> str | None:
        heuristics: dict[str, Iterable[str]] = {
            "python": (r"^\s*def\s", "import ", "pytest", "python"),
            "javascript": ("function(", "console.log", "javascript", "node.js", "npm"),
            "typescript": ("typescript", "ts-node", "tsconfig"),
            "java": ("public class", "System.out.println", "implements", "throws"),
            "cpp": ("std::", "#include <iostream>", "cpp", "c\\+\\+"),
            "c": ("#include <stdio.h>", "int main()", "c programming"),
            "go": ("package main", "fmt.Println", "go build"),
            "rust": ("fn main()", "cargo", "rust", "println!")
        }

        for language, patterns in heuristics.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return language

        known_languages = set(self._normalizer(lang) for lang in available_languages())
        for language in known_languages.union(self.config.fallback_languages):
            if language in text.lower():
                return language

        return None

    def get_last_source(self) -> str:
        return self._last_source
