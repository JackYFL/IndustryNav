"""Render a compact comparison of PointGoal evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=SUMMARY_JSON",
        help="Labeled evaluator summary. Repeat to compare multiple runs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--title", default="PointGoal RL · input_points.json · seed 0")
    return parser.parse_args()


def summarize(label: str, path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    episodes = report["episodes"]
    total_steps = sum(int(row["steps_taken"]) for row in episodes)
    return {
        "label": label,
        "summary": str(path),
        "episodes": len(episodes),
        "successes": sum(bool(row["success"]) for row in episodes),
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "mean_final_distance_m": float(np.mean([
            row["distance_world"] for row in episodes
        ])),
        "collision_step_rate": (
            sum(int(row["collision_steps"]) for row in episodes) / total_steps
        ),
        "warning_step_rate": (
            sum(int(row["warning_steps"]) for row in episodes) / total_steps
        ),
        "mean_steps": float(np.mean([row["steps_taken"] for row in episodes])),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for value in args.run:
        if "=" not in value:
            raise SystemExit(f"--run must be LABEL=SUMMARY_JSON, got: {value}")
        label, raw_path = value.split("=", 1)
        rows.append(summarize(label, Path(raw_path)))

    labels = [row["label"] for row in rows]
    colors = ["#94A3B8", "#38BDF8", "#22C55E"]
    colors = [colors[min(index, len(colors) - 1)] for index in range(len(rows))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    panels = (
        ("success_rate", "Success@2m", "%", 100.0),
        ("mean_final_distance_m", "Mean final distance", "m", 1.0),
        ("collision_step_rate", "Collision-step rate", "%", 100.0),
        ("warning_step_rate", "Warning-step rate", "%", 100.0),
    )
    for axis, (key, title, unit, scale) in zip(axes.flat, panels, strict=True):
        values = [row[key] * scale for row in rows]
        bars = axis.bar(labels, values, color=colors, width=0.62)
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", rotation=10)
        for bar, value in zip(bars, values, strict=True):
            suffix = "%" if unit == "%" else " m"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}{suffix}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes[0, 0].axhline(60.0, color="#EF4444", linestyle="--", linewidth=1.5)
    axes[0, 0].text(
        len(rows) - 0.52,
        60.8,
        "60% target",
        color="#B91C1C",
        fontsize=9,
        ha="right",
    )
    fig.suptitle(args.title, fontsize=15, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)

    metrics_path = args.metrics_json or args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} and {metrics_path}")


if __name__ == "__main__":
    main()
