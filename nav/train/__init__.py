"""Behavior-cloning training + inference.

- :mod:`nav.train.dataset` — episode datasets (single-frame + sequence).
- :mod:`nav.train.loop` — the model-agnostic training loop (:func:`train`).
- :mod:`nav.train.controller` — the inference-time :class:`BCNavController`
  used by the benchmark harness in ``bc_agent`` mode.

``controller`` is imported lazily (it pulls in torch via the policy import);
``from nav.train import BCNavController`` still works.
"""

from nav.train.dataset import NavEpisodeDataset, NavEpisodeSequenceDataset
from nav.train.loop import train

__all__ = [
    "NavEpisodeDataset",
    "NavEpisodeSequenceDataset",
    "train",
    "BCNavController",
]


def __getattr__(name: str):  # lazy re-export to avoid importing the controller eagerly
    if name == "BCNavController":
        from nav.train.controller import BCNavController

        return BCNavController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
