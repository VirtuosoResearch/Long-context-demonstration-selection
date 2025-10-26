"""
Convenience wrappers for launching HumanEval agent inference.
"""

from __future__ import annotations

from typing import Any

from mta.engine.humaneval_runner import HumanEvalAgentRunner, HumanEvalDatasetConfig
from mta.engine.sweagent_vllm import VLLMConnectionConfig


def run_humaneval_agent(
    *,
    model: str,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    engine: str = "openai",
    dataset_path: str | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    languages: list[str] | None = None,
    dataset_name: str | None = None,
    dataset_split: str = "test",
    runner_kwargs: dict[str, Any] | None = None,
) -> list:
    """
    High-level helper that mirrors :class:`HumanEvalAgentRunner`.

    Parameters map directly to the CLI exposed in ``mta/scripts/run_humaneval_agent.py``.
    Additional keyword arguments can be forwarded to the runner constructor via
    ``runner_kwargs``.
    """

    connection = VLLMConnectionConfig(model=model, base_url=base_url, api_key=api_key)
    runner_kwargs = runner_kwargs.copy() if runner_kwargs else {}
    runner_kwargs.setdefault("engine_name", engine)

    runner = HumanEvalAgentRunner(connection=connection, **runner_kwargs)
    dataset_cfg = HumanEvalDatasetConfig()
    if dataset_path:
        dataset_cfg.path = dataset_path
    else:
        if dataset_name:
            dataset_cfg.dataset_name = dataset_name
        dataset_cfg.split = dataset_split
        if languages:
            if len(languages) == 1:
                dataset_cfg.language = languages[0]
                dataset_cfg.languages = None
            else:
                dataset_cfg.languages = languages
                dataset_cfg.language = None
    if limit is not None:
        dataset_cfg.limit = limit
    if task_ids:
        dataset_cfg.task_ids = task_ids

    return runner.run_dataset(dataset_cfg)
