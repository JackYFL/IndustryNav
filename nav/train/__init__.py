"""Algorithm-neutral training infrastructure.

- :mod:`nav.train.base` — reusable trainer lifecycle.
- :mod:`nav.train.advantages` — policy-gradient advantage estimators.
- :mod:`nav.train.initialization` — cross-method checkpoint initialization.

Legacy BC exports remain available here, but new code should use
``nav.baselines.bc`` and ``nav.data`` directly.
"""

from nav.train.base import BaseTrainer, TrainerState

__all__ = [
    "NavEpisodeDataset",
    "NavEpisodeSequenceDataset",
    "train",
    "BCNavController",
    "BaseTrainer",
    "TrainerState",
]


def __getattr__(name: str):
    if name in {"NavEpisodeDataset", "NavEpisodeSequenceDataset"}:
        from nav.data import NavEpisodeDataset, NavEpisodeSequenceDataset

        return {
            "NavEpisodeDataset": NavEpisodeDataset,
            "NavEpisodeSequenceDataset": NavEpisodeSequenceDataset,
        }[name]
    if name == "train":
        from nav.baselines.bc.trainer import train

        return train
    if name == "BCNavController":
        from nav.baselines.bc.agent import BCNavController

        return BCNavController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
