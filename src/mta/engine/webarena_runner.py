"""Runner for executing the WebArena agent with configurable backends."""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency for API-only runs
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

from mta.agents import WebArenaAgent
from mta.engine import AgentExecutionEngine
from mta.engine.sweagent_vllm import VLLMConnectionConfig
from mta.environments import BrowserGymEnv
from mta.openai_chat_tokenizer import OpenAIChatTokenizer

logger = logging.getLogger(__name__)


@dataclass
class WebArenaAgentConfig:
    """Configuration options specific to the WebArena agent."""

    use_fn_calling: bool = False  # reserved for parity with other agents


class WebArenaRunner:
    """Utility for running WebArenaAgent with either OpenAI or local transformers backends."""

    def __init__(
        self,
        *,
        connection: VLLMConnectionConfig,
        engine_name: str = "openai",
        tokenizer_name: str | None = None,
        trust_remote_code: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        env_defaults: dict[str, Any] | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.connection = connection
        self.engine_name = engine_name
        self.env_defaults = env_defaults.copy() if env_defaults else {}
        self.engine_kwargs = engine_kwargs.copy() if engine_kwargs else {}
        self.model_kwargs = model_kwargs.copy() if model_kwargs else {}
        self.device = device

        sampling_params = self._build_sampling_params()
        rollout_engine_args: dict[str, Any] = {}
        rollout_engine = None

        if engine_name == "openai":
            if tokenizer_name:
                logger.warning(
                    "Tokenizer '%s' provided but engine 'openai' delegates to an HTTP API; skipping Hugging Face download.",
                    tokenizer_name,
                )
            if tokenizer_kwargs:
                logger.warning(
                    "tokenizer_kwargs provided but engine 'openai' delegates to an HTTP API; skipping Hugging Face download."
                )
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
            agent_class=WebArenaAgent,
            env_class=BrowserGymEnv,
            agent_args={},
            env_args={},
            engine_name=engine_name,
            tokenizer=self.tokenizer,
            sampling_params=sampling_params,
            rollout_engine=rollout_engine,
            rollout_engine_args=rollout_engine_args,
            **self.engine_kwargs,
        )

    def _build_sampling_params(self) -> dict[str, Any]:
        params = {"temperature": self.connection.temperature, "model": self.connection.model}
        if self.connection.top_p is not None:
            params["top_p"] = self.connection.top_p
        if self.connection.max_tokens is not None:
            params["max_tokens"] = self.connection.max_tokens
        params.update(self.connection.extra_sampling_params)
        return params

    def _prepare_tasks(self, tasks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for task in tasks:
            config = copy.deepcopy(self.env_defaults)
            config.update(task)
            if "env_id" not in config:
                raise ValueError("Each task must include an 'env_id' key for BrowserGymEnv")
            prepared.append(config)
        return prepared

    async def arun(self, tasks: Sequence[dict[str, Any]]):
        prepared = self._prepare_tasks(tasks)
        if not prepared:
            logger.warning("No tasks provided to WebArenaRunner.arun; returning empty list.")
            return []
        return await self.engine.execute_tasks(prepared)

    def run(self, tasks: Sequence[dict[str, Any]]):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("WebArenaRunner.run cannot be called from within an existing event loop; use 'await arun(...)' instead.")

        prepared = self._prepare_tasks(tasks)
        if not prepared:
            logger.warning("No tasks provided to WebArenaRunner.run; returning empty list.")
            return []
        return asyncio.run(self.engine.execute_tasks(prepared))

    def run_env(self, env_id: str, episodes: int = 1, task: dict[str, Any] | None = None):
        base_task = {"env_id": env_id}
        if task:
            base_task["task"] = task
        tasks = [base_task.copy() for _ in range(episodes)]
        return self.run(tasks)
