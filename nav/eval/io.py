"""Shared loaders for per-run output directories.

The post-hoc evaluators all read from the same on-disk layout produced by
``run_headless_benchmark.py``::

    <input_dir>/
        results.csv           # one-row session summary (RESULTS_CSV_FIELDS)
        agent_actions.csv     # per-frame actions + positions
        agent_depth/          # per-frame depth .npy files
        agent_depth/<N>.png   # optional matching depth visualization
        ...

Different ``--baseline`` runs use slightly different directory/CSV prefixes
(``llm_*``, ``astar_*``, ``navid_*``, ``bc_*`` plus legacy ``agent_*`` /
``bc_agent_*`` / ``manual_*``). The candidate prefix lists live in
:mod:`nav.config`; the helpers here hide that variation behind a single
interface so individual metric implementations don't reimplement probing.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import numpy as np

from nav.config import EVAL_ACTIONS_CSV_CANDIDATES, EVAL_DEPTH_DIR_CANDIDATES

#: Candidate names for the depth-frame subdirectory across baselines.
DEPTH_DIR_CANDIDATES: List[str] = EVAL_DEPTH_DIR_CANDIDATES

#: Candidate names for the per-frame actions CSV across baselines.
ACTIONS_CSV_CANDIDATES: List[str] = EVAL_ACTIONS_CSV_CANDIDATES


def find_first_existing(input_dir: Path, names: List[str]) -> Optional[Path]:
    """Return the first ``input_dir / name`` that exists, else None."""
    for name in names:
        path = input_dir / name
        if path.exists():
            return path
    return None


def find_depth_dir(input_dir: Path) -> Optional[Path]:
    """Locate the depth-frame subdirectory for a run, regardless of exec_mode."""
    return find_first_existing(input_dir, DEPTH_DIR_CANDIDATES)


def find_actions_csv(input_dir: Path) -> Optional[Path]:
    """Locate the per-frame actions CSV for a run, regardless of exec_mode."""
    return find_first_existing(input_dir, ACTIONS_CSV_CANDIDATES)


def read_depth_npy(path: Path) -> Optional[np.ndarray]:
    """Load a depth ``.npy`` as float32, returning None on any read failure."""
    try:
        arr = np.load(path)
    except Exception:
        return None
    if arr is None:
        return None
    return arr.astype(np.float32)


def load_action_steps(actions_csv: Path) -> List[int]:
    """Return the sorted list of step indices present in the actions CSV.

    Used to restrict per-frame evaluation to only those steps that actually
    produced an action (frames with missing/malformed step values are
    silently skipped).
    """
    steps: List[int] = []
    with open(actions_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(float(row["step"])))
            except (ValueError, KeyError, TypeError):
                continue
    return steps


def read_latest_results_row(results_csv: Path) -> dict:
    """Return the last row of ``results.csv``, or ``{}`` if the file is empty or absent.

    Each run typically appends exactly one row, but if a directory was
    reused across runs only the most-recent row reflects the current state.
    """
    if not results_csv.exists():
        return {}
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}
