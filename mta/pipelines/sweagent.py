"""
Programmatic helpers for launching SWE-agent style inference.
"""

from __future__ import annotations

from typing import Any

from mta.engine.sweagent_vllm import SweAgentConfig, SweAgentVLLMRunner, VLLMConnectionConfig


def run_sweagent_vllm(
    *,
    model: str,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    dataset: str = "SWE_Bench_Verified",
    split: str = "test",
    limit: int | None = None,
    engine: str = "openai",
    runner_kwargs: dict[str, Any] | None = None,
) -> list:
    """
    Convenience wrapper around :class:`SweAgentVLLMRunner`.

    Parameters mirror the CLI exposed in ``mta/scripts/run_sweagent_vllm.py``.
    Additional keyword arguments are forwarded to the runner constructor via
    ``runner_kwargs``.
    """

    connection = VLLMConnectionConfig(model=model, base_url=base_url, api_key=api_key)
    runner_kwargs = runner_kwargs.copy() if runner_kwargs else {}

    if "agent_config" not in runner_kwargs:
        runner_kwargs["agent_config"] = SweAgentConfig()

    runner_kwargs.setdefault("engine_name", engine)

    runner = SweAgentVLLMRunner(connection=connection, **runner_kwargs)
    return runner.run_dataset(dataset, split, limit=limit)
