"""
Runner for executing HumanEval tasks through the generic AgentExecutionEngine.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

from human_eval.data import HUMAN_EVAL, read_problems

from mta.agents import HumanEvalAgent
from mta.engine.agent_execution_engine import AgentExecutionEngine
from mta.engine.sweagent_vllm import VLLMConnectionConfig
from mta.environments.human_eval import HumanEvalEnv
from mta.openai_chat_tokenizer import OpenAIChatTokenizer
from mta.task_prompts import resolve_prompt
from mta.utils import LanguageDetectionConfig, LanguageDetector

logger = logging.getLogger(__name__)


@dataclass
class HumanEvalAgentConfig:
    """Configuration for the HumanEval agent scaffold."""

    system_prompt: str | None = None


@dataclass
class HumanEvalDatasetConfig:
    """Dataset loading options for HumanEval."""

    path: str = HUMAN_EVAL
    limit: int | None = None
    task_ids: Sequence[str] | None = None


class HumanEvalAgentRunner:
    """
    Helper for launching HumanEval tasks against either an OpenAI-compatible
    endpoint or a local transformers model.
    """

    def __init__(
        self,
        *,
        connection: VLLMConnectionConfig,
        engine_name: str = "openai",
        tokenizer_name: str | None = None,
        trust_remote_code: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        n_parallel_agents: int = 1,
        max_response_length: int = 4096,
        max_prompt_length: int = 2048,
        max_steps: int = 4,
        agent_config: HumanEvalAgentConfig | dict[str, Any] | None = None,
        env_kwargs: dict[str, Any] | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        self.connection = connection
        self.engine_name = engine_name
        self.env_kwargs = env_kwargs.copy() if env_kwargs else {}
        self.agent_kwargs = self._resolve_agent_kwargs(agent_config)
        self.engine_kwargs = engine_kwargs.copy() if engine_kwargs else {}
        self.model_kwargs = model_kwargs.copy() if model_kwargs else {}
        self.device = device

        default_language = self.agent_kwargs.get("default_language", "python")

        if "language_detector" not in self.agent_kwargs:
            det_cfg_override = self.agent_kwargs.pop("language_detection_config", None)
            default_use_heuristics = engine_name != "openai"
            if isinstance(det_cfg_override, LanguageDetectionConfig):
                detection_config = det_cfg_override
            elif isinstance(det_cfg_override, dict):
                cfg_kwargs = {"default_language": default_language, "use_heuristics": default_use_heuristics}
                cfg_kwargs.update(det_cfg_override)
                detection_config = LanguageDetectionConfig(**cfg_kwargs)
            else:
                detection_config = LanguageDetectionConfig(
                    default_language=default_language,
                    use_heuristics=default_use_heuristics,
                )

            connection_kwargs: dict[str, str] = {}
            if engine_name == "openai":
                connection_kwargs = {
                    "base_url": connection.base_url.rstrip("/"),
                    "api_key": connection.api_key,
                }

            language_detector = LanguageDetector(
                engine_name=engine_name,
                connection_kwargs=connection_kwargs,
                model_name=connection.model,
                config=detection_config,
            )

            self.agent_kwargs.setdefault("language_detector", language_detector)

        self.agent_kwargs.setdefault("default_language", default_language)
        self.agent_kwargs.setdefault("task_prompt_resolver", lambda language: resolve_prompt(language))

        sampling_params = self._build_sampling_params()
        rollout_engine_args: dict[str, Any] = {}
        rollout_engine = None

        if engine_name == "openai":
            self.tokenizer = OpenAIChatTokenizer(model_name=connection.model)
            rollout_engine_args = {
                "base_url": connection.base_url.rstrip("/"),
                "api_key": connection.api_key,
            }
        elif engine_name == "transformers":
            if AutoTokenizer is None or AutoModelForCausalLM is None:
                raise ImportError("transformers must be installed to use the local transformers engine.")

            tokenizer_settings = tokenizer_kwargs.copy() if tokenizer_kwargs else {}
            tokenizer_settings.setdefault("trust_remote_code", trust_remote_code)
            tokenizer_id = tokenizer_name or connection.model
            logger.info("Loading tokenizer %s (trust_remote_code=%s)", tokenizer_id, tokenizer_settings.get("trust_remote_code"))
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_settings)

            local_kwargs = self.model_kwargs.copy()
            if self.device:
                if self.device == "auto":
                    local_kwargs.setdefault("device_map", "auto")
                elif self.device != "cpu":
                    local_kwargs.setdefault("device_map", None)
            logger.info("Loading local model %s for transformers engine", connection.model)
            rollout_engine = AutoModelForCausalLM.from_pretrained(connection.model, **local_kwargs)
            rollout_engine.eval()
            if self.device and self.device not in ("auto",):
                if local_kwargs.get("device_map") in (None,):
                    rollout_engine.to(self.device)
            rollout_engine_args = {"device": self.device}
        else:
            raise ValueError(f"Unsupported engine '{engine_name}'")

        self.engine = AgentExecutionEngine(
            agent_class=HumanEvalAgent,
            env_class=HumanEvalEnv,
            agent_args=self.agent_kwargs,
            env_args=self.env_kwargs,
            engine_name=engine_name,
            tokenizer=self.tokenizer,
            sampling_params=sampling_params,
            rollout_engine=rollout_engine,
            rollout_engine_args=rollout_engine_args,
            n_parallel_agents=n_parallel_agents,
            max_response_length=max_response_length,
            max_prompt_length=max_prompt_length,
            max_steps=max_steps,
            **self.engine_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Dataset helpers
    # ------------------------------------------------------------------ #

    def build_tasks(self, problems: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for problem in problems:
            tasks.append(copy.deepcopy(problem))
        return tasks

    def load_dataset(self, config: HumanEvalDatasetConfig | None = None) -> list[dict[str, Any]]:
        cfg = config or HumanEvalDatasetConfig()
        problem_map = read_problems(cfg.path)

        if cfg.task_ids:
            missing = [task_id for task_id in cfg.task_ids if task_id not in problem_map]
            if missing:
                logger.warning("Skipping unknown HumanEval task id(s): %s", ", ".join(missing))
            selected = [problem_map[task_id] for task_id in cfg.task_ids if task_id in problem_map]
        else:
            selected = [problem_map[key] for key in sorted(problem_map.keys())]

        if cfg.limit is not None:
            selected = selected[: cfg.limit]

        logger.info("Loaded %s HumanEval task(s) from %s", len(selected), cfg.path)
        return self.build_tasks(selected)

    # ------------------------------------------------------------------ #
    # Execution helpers
    # ------------------------------------------------------------------ #

    async def arun(self, tasks: Sequence[dict[str, Any]]):
        if not tasks:
            logger.warning("No tasks provided to HumanEvalAgentRunner.arun; returning empty list.")
            return []
        return await self.engine.execute_tasks(list(tasks))

    def run(self, tasks: Sequence[dict[str, Any]]):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("HumanEvalAgentRunner.run cannot be called inside an existing event loop; use `await arun(...)` instead.")

        if not tasks:
            logger.warning("No tasks provided to HumanEvalAgentRunner.run; returning empty list.")
            return []

        return asyncio.run(self.engine.execute_tasks(list(tasks)))

    def run_dataset(self, config: HumanEvalDatasetConfig | None = None):
        tasks = self.load_dataset(config)
        return self.run(tasks)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_sampling_params(self) -> dict[str, Any]:
        params = {"temperature": self.connection.temperature, "model": self.connection.model}
        if self.connection.top_p is not None:
            params["top_p"] = self.connection.top_p
        if self.connection.max_tokens is not None:
            params["max_tokens"] = self.connection.max_tokens
        params.update(self.connection.extra_sampling_params)
        return params

    def _resolve_agent_kwargs(self, agent_config: HumanEvalAgentConfig | dict[str, Any] | None) -> dict[str, Any]:
        if agent_config is None:
            return {}
        if isinstance(agent_config, HumanEvalAgentConfig):
            data = {k: v for k, v in vars(agent_config).items() if v is not None}
            return data
        return agent_config.copy()
