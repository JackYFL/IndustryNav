"""Streaming metric contracts shared by online and offline evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseMetric(ABC, Generic[InputT, OutputT]):
    """Stateful metric with a stable reset/update/compute lifecycle."""

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state."""

    @abstractmethod
    def update(self, value: InputT) -> None:
        """Consume one eligible observation."""

    @abstractmethod
    def compute(self) -> OutputT:
        """Return the current aggregate."""


class BaseEvaluator(ABC, Generic[InputT, OutputT]):
    """Stable contract for evaluating one artifact or live episode."""

    @abstractmethod
    def evaluate(self, value: InputT) -> OutputT:
        """Evaluate one input and return its metric bundle."""


@dataclass(frozen=True)
class RateResult:
    total: int
    triggered: int
    rate: float


class BinaryRateMetric(BaseMetric[bool, RateResult]):
    """Count the fraction of eligible observations that trigger an event."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0
        self.triggered = 0

    def update(self, value: bool) -> None:
        self.total += 1
        self.triggered += int(bool(value))

    def compute(self) -> RateResult:
        return RateResult(
            total=self.total,
            triggered=self.triggered,
            rate=self.triggered / self.total if self.total else 0.0,
        )
