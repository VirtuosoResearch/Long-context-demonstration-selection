from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnv(ABC):
    @property
    def idx(self) -> Any:
        return getattr(self, "_idx", None)

    @idx.setter
    def idx(self, value: Any):
        self._idx = value

    @abstractmethod
    def reset(self) -> tuple[Any, dict]:
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        pass

    def close(self):
        return

    @staticmethod
    @abstractmethod
    def from_dict(info: dict) -> "BaseEnv":
        raise NotImplementedError("Subclasses must implement the 'from_dict' static method.")

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
