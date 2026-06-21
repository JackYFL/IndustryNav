"""Partial analysis pipeline for scene-label-less data.

Used only when the source data lacks (scene, point) identifiers — today
that means the legacy ``experiment_results_v6.csv`` archive. The
statistical machinery is weaker as a result:

- Unpaired permutation instead of paired (no cell pairing possible).
- Episode-level bootstrap instead of scene-clustered (CI is artificially
  tight; flagged in the report as a lower bound).
- Spearman is deferred entirely.

Anything that has scene labels — grid output tree, xlsx-derived
per_run.csv, merged per_run.csv — should go through
:func:`nav.stats.full.run_full_analysis` instead.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from nav.stats.bootstrap import episode_bootstrap_ci
from nav.stats.permutation import unpaired_permutation_pvalue

logger = logging.getLogger(__name__)


def _bootstrap_rows(
    qualifying: Dict[str, Dict[str, np.ndarray]],
    n_boot: int,
    rng: np.random.Generator,
) -> List[dict]:
    """Per-model SR + distance with episode bootstrap 95% CIs."""
    out: List[dict] = []
    for m, d in sorted(qualifying.items(), key=lambda kv: -len(kv[1]["distance"])):
        sr, sr_lo, sr_hi = episode_bootstrap_ci(d["success"], n_boot, rng)
        dm, dm_lo, dm_hi = episode_bootstrap_ci(d["distance"], n_boot, rng)
        out.append({
            "model": m, "n_runs": len(d["distance"]),
            "sr": sr, "sr_ci_lo": sr_lo, "sr_ci_hi": sr_hi,
            "mean_distance_px": dm, "dist_ci_lo": dm_lo, "dist_ci_hi": dm_hi,
        })
    return out


def _permutation_rows(
    qualifying: Dict[str, Dict[str, np.ndarray]],
    n_perm: int,
    rng: np.random.Generator,
) -> List[dict]:
    """Pairwise unpaired permutation tests over distance + success."""
    out: List[dict] = []
    models_sorted = sorted(qualifying.keys())
    for i, ma in enumerate(models_sorted):
        for mb in models_sorted[i + 1:]:
            xa, xb = qualifying[ma]["distance"], qualifying[mb]["distance"]
            ya, yb = qualifying[ma]["success"], qualifying[mb]["success"]
            p_dist = unpaired_permutation_pvalue(xa, xb, n_perm, rng)
            p_succ = unpaired_permutation_pvalue(
                ya.astype(float), yb.astype(float), n_perm, rng
            )
            out.append({
                "model_a": ma, "model_b": mb,
                "n_a": len(xa), "n_b": len(xb),
                "mean_dist_a": float(xa.mean()), "mean_dist_b": float(xb.mean()),
                "p_distance": p_dist,
                "sr_a": float(ya.mean()), "sr_b": float(yb.mean()),
                "p_success": p_succ,
            })
    return out


def _render_report(
    by_model: Dict[str, Dict[str, np.ndarray]],
    qualifying: Dict[str, Dict[str, np.ndarray]],
    rejected: Dict[str, int],
    boot_rows: List[dict],
    perm_rows: List[dict],
    *,
    source_label: str,
    min_n: int,
    n_perm: int,
    n_boot: int,
) -> str:
    total_runs = sum(len(d["distance"]) for d in by_model.values())
    md: List[str] = []
    md.append("# Partial statistical analysis (existing data)\n")
    md.append(f"Source: `{source_label}` ({total_runs} total runs across "
              f"{len(by_model)} models).\n")
    md.append(f"Inclusion threshold: ≥{min_n} runs per model. "
              f"Permutations: {n_perm:,}. Bootstrap resamples: {n_boot:,}.\n")
    md.append(
        "> **Caveats.** The archived CSV does not record scene/point identifiers, so:\n"
        "> - Item (2) below uses an **unpaired** two-sample permutation test on "
        "per-run distance and success (the proper paired version is deferred to the "
        "camera-ready, after the planned re-runs land scene-labeled records).\n"
        "> - Item (4) below uses **non-clustered** episode-level bootstrap. The proper "
        "scene-clustered CI will be wider; values here should be read as a lower bound "
        "on uncertainty.\n"
        "> - Item (3) (Spearman across two independent target sets) cannot be computed "
        "without scene labels and is **deferred** entirely.\n"
    )

    md.append("## (4) Per-model bootstrap CIs (95%)\n")
    md.append("| Model | N | SR | 95% CI (SR) | Mean dist (px) | 95% CI (dist) |\n")
    md.append("|---|---:|---:|---|---:|---|\n")
    for r in boot_rows:
        md.append(f"| `{r['model']}` | {r['n_runs']} | {r['sr']:.3f} | "
                  f"[{r['sr_ci_lo']:.3f}, {r['sr_ci_hi']:.3f}] | "
                  f"{r['mean_distance_px']:.1f} | "
                  f"[{r['dist_ci_lo']:.1f}, {r['dist_ci_hi']:.1f}] |\n")

    md.append("\n## (2) Pairwise unpaired permutation tests\n")
    if perm_rows:
        md.append("Two-sided p-values (additive smoothing prevents p=0):\n\n")
        md.append("| A | B | n_A | n_B | Δmean dist | p (dist) | ΔSR | p (SR) |\n")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in perm_rows:
            md.append(f"| `{r['model_a']}` | `{r['model_b']}` | {r['n_a']} | {r['n_b']} | "
                      f"{r['mean_dist_a']-r['mean_dist_b']:+.1f} | {r['p_distance']:.4f} | "
                      f"{r['sr_a']-r['sr_b']:+.3f} | {r['p_success']:.4f} |\n")
    else:
        md.append(f"No qualifying pairs (need ≥2 models with ≥{min_n} runs).\n")

    md.append("\n## (3) Spearman rank correlation\n")
    md.append("**Deferred.** Cannot construct two independent start-target sets without "
              "scene metadata. Computed from the scene-labeled grid by "
              "`nav.stats.full.run_full_analysis` instead.\n")

    if rejected:
        md.append("\n## Models excluded (n < min_n)\n")
        md.append("| Model | n_runs |\n|---|---:|\n")
        for m, n in sorted(rejected.items(), key=lambda kv: -kv[1]):
            md.append(f"| `{m}` | {n} |\n")
    return "".join(md)


def _write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_partial_analysis(
    by_model: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    *,
    min_n: int = 10,
    n_perm: int = 10_000,
    n_boot: int = 10_000,
    seed: int = 0,
    source_label: str = "by_model",
) -> dict:
    """Run the partial pipeline against the by-model arrays from ``load_experiment_v6_by_model``.

    Writes three files to ``out_dir``: ``bootstrap.csv``, ``permutation.csv``,
    ``report.md``. Returns the in-memory tables.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    qualifying = {m: d for m, d in by_model.items() if len(d["distance"]) >= min_n}
    rejected = {m: len(d["distance"]) for m, d in by_model.items() if m not in qualifying}

    boot_rows = _bootstrap_rows(qualifying, n_boot, rng)
    perm_rows = _permutation_rows(qualifying, n_perm, rng)

    _write_csv(
        out_dir / "bootstrap.csv", boot_rows,
        ["model", "n_runs", "sr", "sr_ci_lo", "sr_ci_hi",
         "mean_distance_px", "dist_ci_lo", "dist_ci_hi"],
    )
    _write_csv(
        out_dir / "permutation.csv", perm_rows,
        ["model_a", "model_b", "n_a", "n_b",
         "mean_dist_a", "mean_dist_b", "p_distance",
         "sr_a", "sr_b", "p_success"],
    )

    report_md = _render_report(
        by_model, qualifying, rejected, boot_rows, perm_rows,
        source_label=source_label, min_n=min_n, n_perm=n_perm, n_boot=n_boot,
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    logger.info(f"[stats.partial] qualifying models (n>={min_n}): {len(qualifying)}")
    for fname in ("report.md", "bootstrap.csv", "permutation.csv"):
        logger.info(f"[stats.partial] wrote {out_dir / fname}")

    return {
        "qualifying": qualifying,
        "rejected": rejected,
        "bootstrap": boot_rows,
        "permutation": perm_rows,
        "report_md": report_md,
    }
