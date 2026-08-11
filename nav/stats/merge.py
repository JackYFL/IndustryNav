"""Supersession-merge of the new-grid + xlsx per_run sources.

The grid records 3 seeds per cell; the xlsx is single-run per cell. To
make pair-matched tests well-defined we collapse the grid's seeds into a
per-cell mean first, then apply a per-metric "newer source supersedes
older source" rule. The provenance trace (which source contributed each
metric for each cell) is written alongside the merged per_run.csv so the
audit trail is preserved.

Caveats — see ``analysis/nav1/additional_details.md`` for the full
write-up. The short version: merged paired tests pair a grid cell against
an xlsx cell at the same (scene, point), which controls for scene
difficulty but **not** for code/prompt/threshold drift between the two
sources. The ``prefer`` flag exists precisely so the rebuttal can flip
the direction when one source is believed corrupted.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from nav.config import STATS_METRIC_FIELDS


def _to_float_or_nan(v) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return float("nan")


def collapse_grid_to_cells(
    grid_rows: List[dict], vision_filter: str = "on"
) -> Dict[Tuple[str, str, str], dict]:
    """Collapse per-seed grid rows into one row per (scene, point, model).

    Each metric is averaged across the 3 seeds (or however many are
    present), with non-finite values dropped before averaging. The
    output row carries an ``n_seeds_collapsed`` field so downstream
    consumers can see how much data each cell summarizes.

    ``vision_filter`` selects which vision condition to fold in: ``"on"``
    (the xlsx is single-condition vision-on, so the merge has to use the
    grid's vision-on rows to be comparable) or ``"off"``.
    """
    bucket: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in grid_rows:
        v_in = str(r.get("vision_input", "True")).lower() in {"1", "true", "yes", "on"}
        if vision_filter == "on" and not v_in:
            continue
        if vision_filter == "off" and v_in:
            continue
        key = (r["scene_name"], r["point_id"], r["model"])
        bucket[key].append(r)

    collapsed: Dict[Tuple[str, str, str], dict] = {}
    for key, rs in bucket.items():
        scene, point, model = key
        first = rs[0]

        def _avg(field: str) -> float:
            vals = [_to_float_or_nan(r.get(field)) for r in rs]
            vals = [v for v in vals if math.isfinite(v)]
            return sum(vals) / len(vals) if vals else float("nan")

        collapsed[key] = {
            "scene_name": scene,
            "point_id": point,
            "model": model,
            "model_short": first.get("model_short") or model.split("/")[-1],
            "vision_input": True if vision_filter == "on" else False,
            "seed_id": "merged",
            "success": _avg("success"),
            "distance_world": _avg("distance_world"),
            "distance_ratio": _avg("distance_ratio"),
            "collision_rate": _avg("collision_rate"),
            "warning_rate": _avg("warning_rate"),
            "efficiency_steps": _avg("efficiency_steps"),
            "steps_taken": _avg("steps_taken"),
            "n_seeds_collapsed": len(rs),
        }
    return collapsed


def index_by_cell(rows: List[dict]) -> Dict[Tuple[str, str, str], dict]:
    """Index xlsx rows by (scene_name, point_id, model). One row per key."""
    return {(r["scene_name"], r["point_id"], r["model"]): dict(r) for r in rows}


def merge(
    grid_collapsed: Dict[Tuple[str, str, str], dict],
    xlsx_indexed: Dict[Tuple[str, str, str], dict],
    prefer: str = "xlsx",
) -> List[dict]:
    """Apply the supersession rule and return merged rows + per-metric provenance.

    Per metric, per cell:

    - First check the preferred source. If the cell exists there and the
      metric is finite, use that value.
    - Else check the other source.
    - Else NaN.

    Each output row carries ``primary_source`` (the table the cell was
    primarily sourced from) and ``<metric>_source`` per metric so the
    per-metric provenance is preserved.
    """
    if prefer not in {"xlsx", "grid"}:
        raise ValueError(f"prefer must be 'xlsx' or 'grid', got {prefer!r}")

    if prefer == "xlsx":
        primary, secondary = xlsx_indexed, grid_collapsed
        p_name, s_name = "xlsx", "grid"
    else:
        primary, secondary = grid_collapsed, xlsx_indexed
        p_name, s_name = "grid", "xlsx"

    all_keys = set(grid_collapsed) | set(xlsx_indexed)
    merged_rows: List[dict] = []

    for key in sorted(all_keys):
        scene, point, model = key
        in_primary = key in primary
        in_secondary = key in secondary
        p_row = primary.get(key, {})
        s_row = secondary.get(key, {})

        out: dict = {
            "scene_name": scene,
            "point_id": point,
            "model": model,
            "model_short": (
                p_row.get("model_short") if in_primary else s_row.get("model_short")
            ) or model.split("/")[-1],
            # Merged rows are always tagged vision_input=True — both inputs to
            # the merge are vision-on by construction.
            "vision_input": True,
            "seed_id": "merged",
            "primary_source": p_name if in_primary else s_name,
        }
        for f in STATS_METRIC_FIELDS:
            pv = _to_float_or_nan(p_row.get(f)) if in_primary else float("nan")
            sv = _to_float_or_nan(s_row.get(f)) if in_secondary else float("nan")
            if in_primary and math.isfinite(pv):
                out[f] = pv
                out[f"{f}_source"] = p_name
            elif in_secondary and math.isfinite(sv):
                out[f] = sv
                out[f"{f}_source"] = s_name
            else:
                out[f] = float("nan")
                out[f"{f}_source"] = "none"
        merged_rows.append(out)
    return merged_rows


def write_merged_outputs(
    merged_rows: List[dict],
    out_path: Path,
) -> Tuple[Path, Path]:
    """Write the merged per_run.csv + a sibling provenance CSV.

    The per_run.csv drops the ``*_source`` columns to stay compatible with
    the canonical loader; the provenance CSV preserves them. Returns the
    two paths written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    main_fields = [
        "scene_name", "point_id", "model", "model_short",
        "vision_input", "seed_id", "primary_source",
    ] + list(STATS_METRIC_FIELDS)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=main_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged_rows)

    prov_path = out_path.parent / "per_run_provenance.csv"
    prov_fields = ["scene_name", "point_id", "model", "primary_source"] + [
        f"{f}_source" for f in STATS_METRIC_FIELDS
    ]
    with open(prov_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=prov_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged_rows)
    return out_path, prov_path
