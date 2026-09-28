"""Common inference contract implemented by learned navigation baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class NavigationAgent(ABC):
    """Minimal lifecycle shared by stateful navigation agents.

    Concrete agents intentionally keep their typed observation arguments: BC
    and PPO consume different sensor bundles today.  The stable contract is
    that an episode starts with :meth:`reset` and every decision returns one
    canonical action string from :meth:`predict_action`.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset recurrent state at an episode boundary."""

    @abstractmethod
    def predict_action(self, *args: Any, **kwargs: Any) -> str:
        """Return the next canonical navigation action."""
