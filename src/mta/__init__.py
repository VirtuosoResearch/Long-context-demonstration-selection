"""
Multi-task agent (mta) package.

This package mirrors the structure of the original rllm project while exposing
a focused surface tailored for SWE-agent style inference pipelines backed by
vLLM.
"""

from .agents import HumanEvalAgent, SWEAgent
from .data import Dataset, DatasetRegistry
from .engine import AgentExecutionEngine, HumanEvalAgentRunner, WebArenaRunner
from .engine.sweagent_vllm import SweAgentVLLMRunner
from .environments import BrowserGymEnv, ConfigurableSWEEnv, HumanEvalEnv, SWEEnv

__all__ = [
    "HumanEvalAgent",
    "SWEAgent",
    "WebArenaAgent",
    "SweAgentVLLMRunner",
    "WebArenaRunner",
    "HumanEvalAgentRunner",
    "AgentExecutionEngine",
    "Dataset",
    "DatasetRegistry",
    "BrowserGymEnv",
    "SWEEnv",
    "ConfigurableSWEEnv",
    "HumanEvalEnv",
]
