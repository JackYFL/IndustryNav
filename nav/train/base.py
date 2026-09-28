"""Reusable trainer lifecycle independent of any navigation algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


ConfigT = TypeVar("ConfigT")
ResultT = TypeVar("ResultT")


@dataclass
class TrainerState:
    """Progress shared by epoch- and update-based trainers."""

    iteration: int = 0
    global_step: int = 0
    best_score: float = float("-inf")


class BaseTrainer(ABC, Generic[ConfigT, ResultT]):
    """Template for setup/fit/teardown with reliable cleanup."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self.state = TrainerState()

    def setup(self) -> None:
        """Allocate resources before training. Optional for subclasses."""

    @abstractmethod
    def fit(self) -> Optional[ResultT]:
        """Execute the algorithm-specific optimization loop."""

    def teardown(self) -> None:
        """Release resources after training. Optional for subclasses."""

    def run(self) -> Optional[ResultT]:
        self.setup()
        try:
            return self.fit()
        finally:
            self.teardown()
