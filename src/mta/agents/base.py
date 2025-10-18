from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Step:
    chat_completions: list[dict[str, str]] = field(default_factory=list)

    thought: str = ""
    action: Any = None
    observation: Any = None
    model_response: str = ""
    info: dict = field(default_factory=dict)

    reward: float = 0.0
    done: bool = False
    mc_return: float = 0.0


@dataclass
class Action:
    action: Any = None


@dataclass
class Trajectory:
    task: Any = None
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [asdict(step) for step in self.steps],
            "reward": float(self.reward),
        }


class BaseAgent(ABC):
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return []

    @property
    def trajectory(self) -> Trajectory:
        return Trajectory()

    @abstractmethod
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_from_model(self, response: Any, **kwargs) -> Action:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        return

    def get_current_state(self) -> Step | None:
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[-1]
