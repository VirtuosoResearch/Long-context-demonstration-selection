"""
Multi-task agent (mta) package.

This package mirrors the structure of the original rllm project while exposing
a focused surface tailored for SWE-agent style inference pipelines backed by
vLLM.
"""

from .agents import SWEAgent
from .data import Dataset, DatasetRegistry
from .engine import AgentExecutionEngine
from .engine.sweagent_vllm import SweAgentVLLMRunner

__all__ = [
    "SWEAgent",
    "SweAgentVLLMRunner",
    "AgentExecutionEngine",
    "Dataset",
    "DatasetRegistry",
]
