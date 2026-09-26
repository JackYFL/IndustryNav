"""Environment contracts shared by train and evaluation workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class NavigationEnvironment(ABC):
    """Minimal Gym-like lifecycle without depending on Gym itself."""

    @abstractmethod
    def reset(self) -> Any:
        """Start the next episode and return its initial observation."""

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Advance the environment by one action."""

    @abstractmethod
    def close(self) -> None:
        """Release external simulator resources."""
