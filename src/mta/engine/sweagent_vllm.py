"""
Utilities for running SWE-agent style inference backed by vLLM.

The runner mirrors the original rllm execution flow while fixing the wiring
for SWE-agent prompts and exposing configuration hooks for an OpenAI-compatible
vLLM server.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency for API-only users
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

from mta.agents import SWEAgent
from mta.data import DatasetRegistry
from mta.engine import AgentExecutionEngine
from mta.environments import ConfigurableSWEEnv, ENV_KWARGS_KEY
from mta.openai_chat_tokenizer import OpenAIChatTokenizer

logger = logging.getLogger(__name__)


@dataclass
class VLLMConnectionConfig:
    """Configuration for connecting to an OpenAI-compatible vLLM server."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 1.0
    top_p: float | None = None
    max_tokens: int | None = None
    extra_sampling_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweAgentConfig:
    """Default settings for the SWEAgent prompt scaffold."""

    use_fn_calling: bool = False
    format_model_response: bool = False
    scaffold: str = "sweagent"


class SweAgentVLLMRunner:
    """
    Helper for launching SWE-agent inference backed by a vLLM OpenAI server.

    Parameters
    ----------
    connection : VLLMConnectionConfig
        Details about the model registered on the vLLM server and sampling choices.
    tokenizer_name : str | None
        Name or path of the tokenizer to load. Defaults to ``connection.model``.
    trust_remote_code : bool
        Whether to enable ``trust_remote_code`` when loading the tokenizer.
    n_parallel_agents : int
        Number of agents that will be scheduled concurrently.
    max_response_length : int
        Maximum length of the sampled response sequence.
    max_prompt_length : int
        Max prompt tokens forwarded to vLLM.
    max_steps : int
        Maximum number of interaction steps per trajectory.
    agent_config : SweAgentConfig | dict[str, Any] | None
        Additional kwargs forwarded to :class:`mta.agents.swe_agent.SWEAgent`.
    env_kwargs : dict[str, Any] | None
        Keyword arguments for the SWE environment (e.g., backend/scaffold).
    engine_kwargs : dict[str, Any]
        Additional overrides for :class:`AgentExecutionEngine`.
    tokenizer_kwargs : dict[str, Any] | None
        Extra kwargs forwarded to :func:`transformers.AutoTokenizer.from_pretrained`.
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
        max_response_length: int = 65536,
        max_prompt_length: int = 4096,
        max_steps: int = 40,
        agent_config: SweAgentConfig | dict[str, Any] | None = None,
        env_kwargs: dict[str, Any] | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        self.connection = connection
        self.engine_name = engine_name
        self.env_kwargs = {"scaffold": "sweagent"}
        if env_kwargs:
            self.env_kwargs.update(env_kwargs)
        self.agent_kwargs = self._resolve_agent_kwargs(agent_config)
        self.engine_kwargs = engine_kwargs.copy() if engine_kwargs else {}
        self.local_model = None
        self.device = device
        self.model_kwargs = model_kwargs.copy() if model_kwargs else {}

        if self.engine_name == "openai":
            if tokenizer_name:
                logger.warning(
                    "Tokenizer '%s' was provided but --engine openai is using the HTTP API; skipping Hugging Face download.",
                    tokenizer_name,
                )
            self.tokenizer = OpenAIChatTokenizer(model_name=connection.model)
        else:
            if AutoTokenizer is None:
                raise ImportError("transformers must be installed to use the local transformers engine")
            tokenizer_settings = tokenizer_kwargs.copy() if tokenizer_kwargs else {}
            tokenizer_settings.setdefault("trust_remote_code", trust_remote_code)
            tokenizer_id = tokenizer_name or connection.model
            logger.info("Loading tokenizer %s (trust_remote_code=%s)", tokenizer_id, tokenizer_settings["trust_remote_code"])
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_settings)

        sampling_params = self._build_sampling_params()
        rollout_engine_args: dict[str, Any] = {}
        rollout_engine = None

        if self.engine_name == "openai":
            rollout_engine_args = {
                "base_url": connection.base_url.rstrip("/"),
                "api_key": connection.api_key,
            }
        elif self.engine_name == "transformers":
            if AutoModelForCausalLM is None:
                raise ImportError("transformers must be installed to use the local transformers engine")
            model_load_kwargs = self.model_kwargs.copy()
            if self.device:
                if self.device == "auto":
                    model_load_kwargs.setdefault("device_map", "auto")
                elif self.device != "cpu":
                    model_load_kwargs.setdefault("device_map", None)
            logger.info("Loading local model %s for transformers engine", connection.model)
            self.local_model = AutoModelForCausalLM.from_pretrained(connection.model, **model_load_kwargs)
            self.local_model.eval()
            if self.device and self.device not in ("auto",):
                if model_load_kwargs.get("device_map") in (None,):
                    self.local_model.to(self.device)
            rollout_engine = self.local_model
            rollout_engine_args = {"device": self.device}
        else:
            raise ValueError(f"Unsupported engine '{self.engine_name}'")

        self.engine = AgentExecutionEngine(
            agent_class=SWEAgent,
            env_class=ConfigurableSWEEnv,
            agent_args=self.agent_kwargs,
            env_args={},
            engine_name=self.engine_name,
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

    def _resolve_agent_kwargs(self, agent_config: SweAgentConfig | dict[str, Any] | None) -> dict[str, Any]:
        if agent_config is None:
            return dict(vars(SweAgentConfig()))
        if isinstance(agent_config, SweAgentConfig):
            return dict(vars(agent_config))
        merged = dict(vars(SweAgentConfig()))
        merged.update(agent_config)
        return merged

    def _build_sampling_params(self) -> dict[str, Any]:
        params = {"temperature": self.connection.temperature, "model": self.connection.model}
        if self.connection.top_p is not None:
            params["top_p"] = self.connection.top_p
        if self.connection.max_tokens is not None:
            params["max_tokens"] = self.connection.max_tokens
        params.update(self.connection.extra_sampling_params)
        return params

    # ------------------------------------------------------------------ #
    # Task helpers
    # ------------------------------------------------------------------ #

    def build_tasks(self, entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Wrap raw dataset entries with environment kwargs for the engine."""
        wrapped = []
        for entry in entries:
            payload = {
                "entry": copy.deepcopy(entry),
                ENV_KWARGS_KEY: copy.deepcopy(self.env_kwargs),
            }
            wrapped.append(payload)
        return wrapped

    def load_dataset(self, name: str, split: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Load tasks from the dataset registry and prepare them for execution."""
        if not DatasetRegistry.dataset_exists(name, split):
            msg = f"Dataset '{name}' split '{split}' not found in the registry. Did you run prepare_swe_data.py?"
            raise FileNotFoundError(msg)

        dataset = DatasetRegistry.load_dataset(name, split)
        if dataset is None:
            raise FileNotFoundError(f"Dataset '{name}' split '{split}' could not be loaded from the registry.")
        data = dataset.get_data()
        if limit is not None:
            data = data[:limit]
        logger.info("Loaded %s/%s tasks from %s:%s", len(data), len(dataset), name, split)
        return self.build_tasks(data)

    # ------------------------------------------------------------------ #
    # Execution helpers
    # ------------------------------------------------------------------ #

    async def arun(self, tasks: Sequence[dict[str, Any]]):
        """Asynchronously execute a list of prepared tasks."""
        if not tasks:
            logger.warning("No tasks provided to SweAgentVLLMRunner.arun; returning empty list.")
            return []
        return await self.engine.execute_tasks(list(tasks))

    def run(self, tasks: Sequence[dict[str, Any]]):
        """Run the prepared tasks and return completed trajectories."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            msg = "SweAgentVLLMRunner.run cannot be called inside an existing event loop; use `await arun(...)` instead."
            raise RuntimeError(msg)

        return asyncio.run(self.arun(tasks))

    def run_dataset(self, name: str, split: str, limit: int | None = None):
        """Load tasks from the registry and execute them synchronously."""
        tasks = self.load_dataset(name, split, limit=limit)
        return self.run(tasks)
