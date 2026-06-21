"""Prompt templates for the runtime agent harness.

This package both *holds* the prompt ``.txt`` files and *exposes* a typed
loader so the rest of the codebase doesn't have to push around stringly-typed
file paths. Use :class:`PromptName` for canonical references.

Example::

    from nav.prompts import PromptName, load
    template = load(PromptName.EGO_STATE_HISTORY)

For CLI compatibility where an arbitrary user-supplied path is accepted
(``--prompt_file <path>``), fall back to
:func:`nav.utils.load_prompt_template`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

#: Absolute path to this package's directory. All shipped prompt ``.txt``
#: files live directly underneath; ``path_of(name)`` resolves against this.
PROMPTS_DIR: Path = Path(__file__).resolve().parent


class PromptName(str, Enum):
    """Canonical identifiers for the shipped prompt templates.

    Each value is the basename (without ``.txt``) of the file under
    :data:`PROMPTS_DIR`. Adding a new prompt means: add a ``.txt`` to this
    directory, add an enum member here, optionally add a default-choice
    constant to :mod:`nav.config`.
    """

    EGO_MINIMAP = "nav_ego_minimap"
    EGO_STATE = "nav_ego_state"
    EGO_STATE_HISTORY = "nav_ego_state_history"
    MINIMAP_ONLY = "nav_minimap_only"
    STATE_HISTORY_NO_VISION = "nav_state_history_no_vision"


def path_of(name: PromptName) -> Path:
    """Return the on-disk path of the prompt template ``name``."""
    return PROMPTS_DIR / f"{name.value}.txt"


def load(name: PromptName) -> str:
    """Return the contents of the prompt template ``name``."""
    return path_of(name).read_text(encoding="utf-8")


__all__ = ["PROMPTS_DIR", "PromptName", "path_of", "load"]
