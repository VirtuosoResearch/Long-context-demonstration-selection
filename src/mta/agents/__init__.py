"""
Agent interfaces exposed by the mta package.

The core dataclasses mirror the original rllm equivalents while remaining
self-contained within this package.
"""

from .base import Action, BaseAgent, Step, Trajectory
from .human_eval_agent import HumanEvalAgent
from .swe_agent import SWEAgent
from .webarena_agent import WebArenaAgent

__all__ = ["Action", "BaseAgent", "Step", "Trajectory", "HumanEvalAgent", "SWEAgent", "WebArenaAgent"]
