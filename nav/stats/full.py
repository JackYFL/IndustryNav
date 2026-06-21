"""Full analysis pipeline: scene-clustered bootstrap + paired permutation + Spearman.

Used when the per-run rows carry scene labels (grid output tree, xlsx-derived
per_run.csv, or merged per_run.csv). The "partial" pipeline in
:mod:`nav.stats.partial` is the fallback for label-less sources.

Outputs (all under ``out_dir``):

- ``per_run.csv``     — flat dump of the input rows (transparency).
- ``bootstrap.csv``   — per-(model, vision) point-estimate + 95% CI per metric.
- ``variance.csv``    — per-(model, vision) cell-level mean/var/std per metric.
- ``paired_perm.csv`` — per model-pair, two comparison axes: across-models
                        at fixed vision, and within-model across vision.
- ``spearman.csv``    — leaderboard rank correlation across a point-subset split.
- ``report.md``       — human-readable summary that knits the four together.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from nav.config import STATS_REPORT_BASE_METRICS, STATS_REPORT_OPTIONAL_METRICS
from nav.stats.bootstrap import clustered_bootstrap_ci
from nav.stats.permutation import paired_permutation_pvalue
from nav.stats.spearman import spearman_corr

logger = logging.getLogger(__name__)


def _active_metrics(rows: Sequence[dict]) -> List[Tuple[str, str, str]]:
    """Pick which metrics are computable for this source.

    Always returns the base triplet (SR/DR/CR). WR is only present in
    xlsx-derived data (grid runs don't save raw depth). ``distance_px``
    is only present for grid-tree-walked rows (xlsx leaves it as NaN).
    """
    metrics = list(STATS_REPORT_BASE_METRICS)
    for src_field, triple in STATS_REPORT_OPTIONAL_METRICS.items():
        if any(np.isfinite(r.get(src_field, float("nan"))) for r in rows):
            metrics.append(triple)
    return metrics


def _group_keys(rows: Sequence[dict]) -> List[Tuple[str, bool]]:
    """Return sorted ``(model_short, vision_input)`` keys present in the data."""
    return sorted({(r["model_short"], r["vision_input"]) for r in rows})


def _rows_for(rows: Sequence[dict], model_short: str, vision: bool) -> List[dict]:
    return [r for r in rows if r["model_short"] == model_short and r["vision_input"] == vision]


# ---------------------------------------------------------------------------
# Stage 1 — bootstrap CIs + cell-level variance
# ---------------------------------------------------------------------------

def _bootstrap_rows(
    rows: Sequence[dict],
    keys: Sequence[Tuple[str, bool]],
    metrics: Sequence[Tuple[str, str, str]],
    n_boot: int,
    rng: np.random.Generator,
) -> List[dict]:
    """One row per (model, vision) with point-estimate + CI for each metric."""
    out: List[dict] = []
    for model_short, vision in keys:
        rs = _rows_for(rows, model_short, vision)
        n_scenes_any = len({r["scene_name"] for r in rs})
        row = {
            "model": model_short, "vision_input": vision,
            "n_runs": len(rs), "n_scenes": n_scenes_any,
        }
        for short_key, src_field, _ in metrics:
            per_scene: dict[str, list] = {}
            for r in rs:
                v = r.get(src_field)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    continue
                per_scene.setdefault(r["scene_name"], []).append(float(v))
            per_scene_arr = {k: np.asarray(v) for k, v in per_scene.items()}
            pe, lo, hi = clustered_bootstrap_ci(per_scene_arr, n_boot, rng)
            row[short_key] = pe
            row[f"{short_key}_ci_lo"] = lo
            row[f"{short_key}_ci_hi"] = hi
        out.append(row)
    return out


def _variance_rows(
    rows: Sequence[dict],
    keys: Sequence[Tuple[str, bool]],
    metrics: Sequence[Tuple[str, str, str]],
) -> List[dict]:
    """Pooled mean/var/std per (model, vision) cell observation."""
    out: List[dict] = []
    for model_short, vision in keys:
        rs = _rows_for(rows, model_short, vision)
        row = {"model": model_short, "vision_input": vision, "n_cells": len(rs)}
        for short_key, src_field, _ in metrics:
            vals = np.array([float(r.get(src_field, np.nan)) for r in rs], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 1:
                row[f"{short_key}_mean"] = float(vals.mean())
                row[f"{short_key}_var"] = float(vals.var(ddof=1))
                row[f"{short_key}_std"] = float(vals.std(ddof=1))
            else:
                for suffix in ("mean", "var", "std"):
                    row[f"{short_key}_{suffix}"] = float("nan")
            row[f"{short_key}_n"] = int(len(vals))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Stage 2 — paired permutation
# ---------------------------------------------------------------------------

def _paired_perm_one(
    group_a_rows: Sequence[dict],
    group_b_rows: Sequence[dict],
    metrics: Sequence[Tuple[str, str, str]],
    n_perm: int,
    rng: np.random.Generator,
) -> dict:
    """Pair rows on (scene, point, seed) and run per-metric sign-flip test."""
    ra = {(r["scene_name"], r["point_id"], r["seed_id"]): r for r in group_a_rows}
    rb = {(r["scene_name"], r["point_id"], r["seed_id"]): r for r in group_b_rows}
    shared = sorted(set(ra) & set(rb))
    result: dict = {"n_paired_cells": len(shared)}
    for short_key, src_field, _ in metrics:
        diffs: List[float] = []
        for k in shared:
            va, vb = ra[k].get(src_field), rb[k].get(src_field)
            if va is None or vb is None:
                continue
            if isinstance(va, float) and not np.isfinite(va):
                continue
            if isinstance(vb, float) and not np.isfinite(vb):
                continue
            diffs.append(float(va) - float(vb))
        if not diffs:
            result[f"mean_diff_{short_key}"] = float("nan")
            result[f"p_{short_key}_paired"] = float("nan")
            continue
        arr = np.asarray(diffs)
        result[f"mean_diff_{short_key}"] = float(arr.mean())
        result[f"p_{short_key}_paired"] = paired_permutation_pvalue(arr, n_perm, rng)
    return result


def _permutation_rows(
    rows: Sequence[dict],
    keys: Sequence[Tuple[str, bool]],
    metrics: Sequence[Tuple[str, str, str]],
    n_perm: int,
    rng: np.random.Generator,
) -> List[dict]:
    """Cross-models-at-fixed-vision + within-model-across-vision comparison rows."""
    out: List[dict] = []

    # (a) across models, fixed vision setting
    for vision in sorted({k[1] for k in keys}):
        models_here = sorted({k[0] for k in keys if k[1] == vision})
        for i, ma in enumerate(models_here):
            for mb in models_here[i + 1:]:
                stat = _paired_perm_one(
                    _rows_for(rows, ma, vision),
                    _rows_for(rows, mb, vision),
                    metrics, n_perm, rng,
                )
                if stat["n_paired_cells"] == 0:
                    continue
                out.append({
                    "comparison_kind": "across_models",
                    "axis_a": ma, "axis_b": mb,
                    "vision_a": "on" if vision else "off",
                    "vision_b": "on" if vision else "off",
                    **stat,
                })

    # (b) within model, vision on vs off (modality ablation)
    for m in sorted({k[0] for k in keys}):
        if not (any(k == (m, True) for k in keys) and any(k == (m, False) for k in keys)):
            continue
        stat = _paired_perm_one(
            _rows_for(rows, m, True),
            _rows_for(rows, m, False),
            metrics, n_perm, rng,
        )
        if stat["n_paired_cells"] == 0:
            continue
        out.append({
            "comparison_kind": "modality_ablation",
            "axis_a": m, "axis_b": m,
            "vision_a": "on", "vision_b": "off",
            **stat,
        })
    return out


# ---------------------------------------------------------------------------
# Stage 3 — Spearman leaderboard correlation
# ---------------------------------------------------------------------------

def _spearman_rows(
    rows: Sequence[dict],
    keys: Sequence[Tuple[str, bool]],
    split: str,
) -> List[dict]:
    """Per-vision rank correlation between two point-subset leaderboards."""
    out: List[dict] = []
    for vision in sorted({k[1] for k in keys}):
        models_here = sorted({k[0] for k in keys if k[1] == vision})
        if len(models_here) < 2:
            continue

        scenes_to_points: dict[str, list[str]] = {}
        for r in rows:
            if r["vision_input"] != vision:
                continue
            scenes_to_points.setdefault(r["scene_name"], []).append(r["point_id"])
        scenes_to_points = {s: sorted(set(p)) for s, p in scenes_to_points.items()}

        set_a: set[Tuple[str, str]] = set()
        set_b: set[Tuple[str, str]] = set()
        for s, pts in scenes_to_points.items():
            if split == "halves":
                mid = len(pts) // 2
                for p in pts[:mid]:
                    set_a.add((s, p))
                for p in pts[mid:]:
                    set_b.add((s, p))
            elif split == "odd_even":
                for i, p in enumerate(pts):
                    (set_a if i % 2 == 0 else set_b).add((s, p))
            else:
                raise ValueError(f"unknown spearman_split: {split!r}")

        def _leaderboard_sr(model_short: str, target: set) -> float:
            rs = [r for r in _rows_for(rows, model_short, vision)
                  if (r["scene_name"], r["point_id"]) in target]
            return float(np.mean([r["success"] for r in rs])) if rs else float("nan")

        sr_a = np.array([_leaderboard_sr(m, set_a) for m in models_here])
        sr_b = np.array([_leaderboard_sr(m, set_b) for m in models_here])
        rho = spearman_corr(sr_a, sr_b) if len(models_here) >= 3 else float("nan")
        out.append({
            "vision_input": vision,
            "split": split,
            "models": ";".join(models_here),
            "n_models": len(models_here),
            "leaderboard_set_a": ";".join(f"{m}={v:.3f}" for m, v in zip(models_here, sr_a)),
            "leaderboard_set_b": ";".join(f"{m}={v:.3f}" for m, v in zip(models_here, sr_b)),
            "spearman_rho": rho,
        })
    return out


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _render_report(
    rows: Sequence[dict],
    keys: Sequence[Tuple[str, bool]],
    metrics: Sequence[Tuple[str, str, str]],
    boot_rows: Sequence[dict],
    variance_rows: Sequence[dict],
    perm_rows: Sequence[dict],
    spearman_rows: Sequence[dict],
    *,
    source_label: str,
    n_perm: int,
    n_boot: int,
    spearman_split: str,
) -> str:
    has_wr = any(short == "wr" for short, _, _ in metrics)
    has_dist = any(short == "dist" for short, _, _ in metrics)

    md: List[str] = []
    md.append("# Full statistical analysis\n\n")
    md.append(f"Source: `{source_label}` ({len(rows)} runs total). ")
    md.append(f"Permutations: {n_perm:,}. Bootstrap resamples: {n_boot:,}. "
              f"Spearman split: `{spearman_split}`.\n\n")
    md.append("**Metric definitions** (matching `eval_metrics.py`):\n")
    md.append("- **SR** (success rate): final `distance_px < 65`. Per-cell binary 0/1.\n")
    md.append("- **DR** (distance ratio): `|start_dist − final_dist| / start_dist`, clamped to 1 on success.\n")
    md.append("- **CR** (collision rate): forward-action steps with Manhattan pixel change < 34, divided by total forward-action steps.\n")
    if has_wr:
        md.append("- **WR** (warning rate): fraction of depth frames with min ROI depth below the warning threshold.\n")
    else:
        md.append("- **WR** (warning rate): **DEFERRED** — grid runs did not save raw depth NPYs. Re-enable NPY saving to recover this.\n")
    if has_dist:
        md.append("- **Mean dist (px)**: average final `distance_px` per run.\n")
    md.append("\n")
    md.append("> **CR caveat.** With the `eval_metrics.py` collision-rate definition, "
              "CR can sit near ceiling for some model/condition combos — the 34 px "
              "threshold is tight relative to the per-step movement budget. CR is "
              "reported for transparency; threshold tuning is queued for the camera-ready.\n\n")

    md.append("## (4) Per-(model, vision) scene-clustered bootstrap CIs (95%)\n")
    hdr = "| Model | Vision | N runs | N scenes "
    sep = "|---|:---:|---:|---:"
    for _, _, label in metrics:
        hdr += f"| {label} | 95% CI "
        sep += "|---:|---"
    hdr += "|\n"; sep += "|\n"
    md.append(hdr); md.append(sep)
    for r in boot_rows:
        line = (f"| `{r['model']}` | {'on' if r['vision_input'] else 'off'} | "
                f"{r['n_runs']} | {r['n_scenes']} ")
        for short_key, _, _ in metrics:
            pe = r.get(short_key, float("nan"))
            lo = r.get(f"{short_key}_ci_lo", float("nan"))
            hi = r.get(f"{short_key}_ci_hi", float("nan"))
            fmt = "{:.1f}" if short_key == "dist" else "{:.3f}"
            line += f"| {fmt.format(pe)} | [{fmt.format(lo)}, {fmt.format(hi)}] "
        line += "|\n"
        md.append(line)

    md.append("\n## (4b) Per-(model, vision) cell-level variance (pooled over seeds × scenes × points)\n")
    md.append("Sample variance (ddof=1) and standard deviation of each per-cell metric, "
              "pooled across (scene, point, seed) cells. Different from the bootstrap CI: "
              "CI is uncertainty of the *mean*, this is the spread of *individual cells*.\n\n")
    hdr = "| Model | Vision | N cells "
    sep = "|---|:---:|---:"
    for _, _, label in metrics:
        hdr += f"| {label} mean | {label} var | {label} std "
        sep += "|---:|---:|---:"
    hdr += "|\n"; sep += "|\n"
    md.append(hdr); md.append(sep)
    for r in variance_rows:
        line = f"| `{r['model']}` | {'on' if r['vision_input'] else 'off'} | {r['n_cells']} "
        for short_key, _, _ in metrics:
            mean = r.get(f"{short_key}_mean", float("nan"))
            var = r.get(f"{short_key}_var", float("nan"))
            std = r.get(f"{short_key}_std", float("nan"))
            if short_key == "dist":
                line += f"| {mean:.1f} | {var:.1f} | {std:.1f} "
            else:
                line += f"| {mean:.3f} | {var:.4f} | {std:.3f} "
        line += "|\n"
        md.append(line)

    md.append("\n## (2a) Paired permutation: across models, fixed vision setting\n")
    md.append("Two-sided p-values from sign-flip on per-cell differences matched on "
              "`(scene, point, seed)`. Δ = A − B.\n\n")
    cross = [r for r in perm_rows if r["comparison_kind"] == "across_models"]
    if cross:
        hdr = "| Vision | A | B | N cells "
        sep = "|:---:|---|---|---:"
        for _, _, label in metrics:
            hdr += f"| Δ{label} | p({label}) "
            sep += "|---:|---:"
        hdr += "|\n"; sep += "|\n"
        md.append(hdr); md.append(sep)
        for r in cross:
            line = (f"| {r['vision_a']} | `{r['axis_a']}` | `{r['axis_b']}` | "
                    f"{r['n_paired_cells']} ")
            for short_key, _, _ in metrics:
                d = r.get(f"mean_diff_{short_key}", float("nan"))
                p = r.get(f"p_{short_key}_paired", float("nan"))
                fmt_d = f"{d:+.1f}" if short_key == "dist" else f"{d:+.3f}"
                line += f"| {fmt_d} | {p:.4f} "
            line += "|\n"
            md.append(line)
    else:
        md.append("No model pairs share enough (scene, point, seed) cells for a paired test.\n")

    md.append("\n## (2b) Paired permutation: within model, vision on vs off (modality ablation)\n")
    md.append("A = vision-on, B = vision-off. **Positive ΔSR/ΔDR ⇒ vision helps**; "
              "**negative Δdist/ΔCR ⇒ vision helps**. Significance at p < 0.05.\n\n")
    modality = [r for r in perm_rows if r["comparison_kind"] == "modality_ablation"]
    if modality:
        hdr = "| Model | N cells "
        sep = "|---|---:"
        for _, _, label in metrics:
            hdr += f"| Δ{label} | p({label}) "
            sep += "|---:|---:"
        hdr += "|\n"; sep += "|\n"
        md.append(hdr); md.append(sep)
        for r in modality:
            line = f"| `{r['axis_a']}` | {r['n_paired_cells']} "
            for short_key, _, _ in metrics:
                d = r.get(f"mean_diff_{short_key}", float("nan"))
                p = r.get(f"p_{short_key}_paired", float("nan"))
                fmt_d = f"{d:+.1f}" if short_key == "dist" else f"{d:+.3f}"
                line += f"| {fmt_d} | {p:.4f} "
            line += "|\n"
            md.append(line)
    else:
        visions_present = sorted({k[1] for k in keys})
        if len(visions_present) < 2:
            mode = "vision-on" if visions_present == [True] else "vision-off"
            md.append(f"**N/A.** This data source contains only {mode} cells.\n")
        else:
            md.append("No (model, vision-on, vision-off) cells share matching (scene, point, seed).\n")

    md.append("\n## (3) Leaderboard Spearman rank correlation across two pair subsets\n")
    if spearman_rows:
        md.append("| Vision | Split | N models | Spearman ρ |\n|:---:|:---:|---:|---:|\n")
        for r in spearman_rows:
            rho_disp = "n/a" if (r["spearman_rho"] != r["spearman_rho"]) else f"{r['spearman_rho']:.3f}"
            md.append(f"| {'on' if r['vision_input'] else 'off'} | `{r['split']}` | "
                      f"{r['n_models']} | {rho_disp} |\n")
        if any(r["n_models"] < 3 for r in spearman_rows):
            md.append("\n**Note:** Spearman ρ is degenerate at N=2 models (±1 only). "
                      "Reported as `n/a`. Run more models to make it meaningful.\n")
        md.append("\nDetailed per-model leaderboards are in `spearman.csv`.\n")
    else:
        md.append("Not enough models per vision setting to compute a leaderboard correlation.\n")
    return "".join(md)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_full_analysis(
    rows: List[dict],
    out_dir: Path,
    *,
    n_perm: int = 10_000,
    n_boot: int = 10_000,
    seed: int = 0,
    spearman_split: str = "halves",
    source_label: str = "rows",
) -> dict:
    """Run the full pipeline against ``rows`` and write five files to ``out_dir``.

    Returns a dict of the in-memory tables produced so callers (e.g. tests)
    don't have to re-read the written CSVs to inspect results.
    """
    if not rows:
        raise ValueError("run_full_analysis called with empty rows")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    keys = _group_keys(rows)
    metrics = _active_metrics(rows)

    _write_csv(out_dir / "per_run.csv", rows, list(rows[0].keys()))

    boot_rows = _bootstrap_rows(rows, keys, metrics, n_boot, rng)
    boot_fields = ["model", "vision_input", "n_runs", "n_scenes"]
    for short_key, _, _ in metrics:
        boot_fields += [short_key, f"{short_key}_ci_lo", f"{short_key}_ci_hi"]
    _write_csv(out_dir / "bootstrap.csv", boot_rows, boot_fields)

    variance_rows = _variance_rows(rows, keys, metrics)
    variance_fields = ["model", "vision_input", "n_cells"]
    for short_key, _, _ in metrics:
        variance_fields += [f"{short_key}_n", f"{short_key}_mean",
                            f"{short_key}_var", f"{short_key}_std"]
    _write_csv(out_dir / "variance.csv", variance_rows, variance_fields)

    perm_rows = _permutation_rows(rows, keys, metrics, n_perm, rng)
    perm_fields = ["comparison_kind", "axis_a", "axis_b", "vision_a", "vision_b",
                   "n_paired_cells"]
    for short_key, _, _ in metrics:
        perm_fields += [f"mean_diff_{short_key}", f"p_{short_key}_paired"]
    _write_csv(out_dir / "paired_perm.csv", perm_rows, perm_fields)

    spearman_rows = _spearman_rows(rows, keys, spearman_split)
    _write_csv(out_dir / "spearman.csv", spearman_rows,
               ["vision_input", "split", "models", "n_models",
                "leaderboard_set_a", "leaderboard_set_b", "spearman_rho"])

    report_md = _render_report(
        rows, keys, metrics, boot_rows, variance_rows, perm_rows, spearman_rows,
        source_label=source_label,
        n_perm=n_perm, n_boot=n_boot, spearman_split=spearman_split,
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    logger.info(f"[stats.full] loaded {len(rows)} runs across {len(keys)} (model, vision) groups.")
    for fname in ("report.md", "per_run.csv", "bootstrap.csv", "variance.csv",
                  "paired_perm.csv", "spearman.csv"):
        logger.info(f"[stats.full] wrote {out_dir / fname}")

    return {
        "rows": rows,
        "bootstrap": boot_rows,
        "variance": variance_rows,
        "paired_perm": perm_rows,
        "spearman": spearman_rows,
        "report_md": report_md,
    }
