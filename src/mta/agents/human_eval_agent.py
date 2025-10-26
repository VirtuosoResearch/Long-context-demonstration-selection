from __future__ import annotations

import logging
from typing import Any, Callable

from mta.agents.base import Action, BaseAgent, Step, Trajectory
from mta.agents.system_prompts import HUMAN_EVAL_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class HumanEvalAgent(BaseAgent):
    """Lightweight agent scaffold for HumanEval-style single-tool interactions."""

    def __init__(
        self,
        system_prompt: str = HUMAN_EVAL_SYSTEM_PROMPT,
        language_detector: Any | None = None,
        task_prompt_resolver: Callable[[str], str | None] | None = None,
        default_language: str = "python",
    ):
        self.system_prompt = system_prompt
        self.language_detector = language_detector
        self.task_prompt_resolver = task_prompt_resolver
        self.default_language = default_language

        self._trajectory = Trajectory()
        self._messages: list[dict[str, str]] = []
        self._cur_step: Step | None = None
        self._step_count = 0
        self._language_prompt_inserted = False
        self._detected_language: str | None = None
        self._detected_language_source: str | None = None

        self.reset()

    # ------------------------------------------------------------------ #
    # BaseAgent API
    # ------------------------------------------------------------------ #

    def reset(self):
        self._trajectory = Trajectory()
        self._messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self._cur_step = None
        self._step_count = 0
        self._language_prompt_inserted = False
        self._detected_language = None
        self._detected_language_source = None

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        observation_text = str(observation)
        self._ensure_language_prompt(observation_text, info)
        if info is not None and self._detected_language:
            info.setdefault("language", self._detected_language)
            info.setdefault("detected_language", self._detected_language)
            if self._detected_language_source:
                info.setdefault("detected_language_source", self._detected_language_source)

        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation_text
            prior_step.reward = reward
            prior_step.done = done
            if info:
                prior_step.info.update(info)

        self._messages.append(
            {
                "role": "user",
                "content": observation_text,
            }
        )
        self._cur_step = Step(observation=observation_text)

    def update_from_model(self, response: Any, **kwargs) -> Action:
        if self._cur_step is None:
            self._cur_step = Step()

        self._trajectory.steps.append(self._cur_step)

        response_text = self._extract_response_text(response)

        current_step = self._trajectory.steps[-1]
        current_step.model_response = response
        current_step.thought = response_text
        current_step.action = response_text

        self._messages.append({"role": "assistant", "content": response_text})
        self._step_count += 1
        self._cur_step = None

        return Action(action=response_text)

    def get_current_state(self) -> Step | None:
        if self._trajectory.steps:
            return self._trajectory.steps[-1]
        return self._cur_step

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self._messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response

        choices = getattr(response, "choices", None)
        if choices:
            try:
                first_choice = choices[0]
            except (IndexError, TypeError):
                first_choice = None

            if first_choice is not None:
                message = getattr(first_choice, "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    if content:
                        return content
                text = getattr(first_choice, "text", None)
                if text:
                    return text

        content = getattr(response, "content", None)
        if isinstance(content, str) and content:
            return content

        return str(response)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_language_prompt(self, observation: str, info: dict | None):
        if self._language_prompt_inserted:
            return

        detected_language, source = self._detect_language(observation, info)
        self._detected_language = detected_language
        self._detected_language_source = source

        logger.debug("HumanEvalAgent detected language '%s' via %s", detected_language, source)

        if not self.task_prompt_resolver:
            self._language_prompt_inserted = True
            return

        prompt = self.task_prompt_resolver(detected_language)
        if not prompt:
            self._language_prompt_inserted = True
            return

        # Insert language-specific guidance as an extra system message just after the default one.
        self._messages.insert(
            1,
            {
                "role": "system",
                "content": prompt,
            },
        )
        self._language_prompt_inserted = True

    def _detect_language(self, observation: str, info: dict | None) -> tuple[str, str]:
        if info:
            info_language = info.get("language") or info.get("detected_language")
            if info_language:
                language_str = str(info_language).lower()
                return language_str, "environment"

        if self.language_detector is None:
            return self.default_language, "default"

        try:
            language = self.language_detector.detect(observation)
            if language:
                source = getattr(self.language_detector, "get_last_source", lambda: "unknown")()
                if not source:
                    source = "unknown"
                return str(language).lower(), source
        except Exception:
            pass

        return self.default_language, "default"
