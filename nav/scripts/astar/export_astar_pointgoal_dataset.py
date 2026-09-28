"""Export successful A* collection episodes into the BC dataset contract.

The benchmark runner saves observation frame ``s-1`` immediately before the
expert action logged as step ``s``.  This exporter makes that relationship
explicit and rewrites both to a common zero-based decision index.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from nav.config import BC_ACTION_TO_LABEL


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-failed", action="store_true",
                        help="Export terminal failed episodes too (off for expert BC).")
    return parser.parse_args()


def is_success(result: dict) -> bool:
    try:
        distance = float(result.get("distance_world", "inf"))
    except (TypeError, ValueError):
        return False
    return result.get("stop_reason") in {"astar_stop", "reached_vicinity"} and distance <= 2.0


def export_episode(task: dict, raw_dir: Path, destination: Path) -> tuple[int, Counter]:
    action_rows = read_rows(raw_dir / "astar_actions.csv")
    if not action_rows:
        raise ValueError("missing astar_actions.csv rows")
    rgb_source = raw_dir / "astar_fp"
    depth_source = raw_dir / "astar_depth"
    rgb_dest = destination / "keyboard_fp"
    depth_dest = destination / "keyboard_depth"
    rgb_dest.mkdir(parents=True)
    depth_dest.mkdir(parents=True)

    output_rows = []
    action_counts = Counter()
    for output_step, row in enumerate(action_rows):
        action = str(row.get("action", "")).strip().lower()
        if action not in BC_ACTION_TO_LABEL:
            raise ValueError(f"non-policy action {action!r}")
        source_action_step = int(float(row["step"]))
        source_frame_step = source_action_step - 1
        if source_frame_step < 0:
            raise ValueError(f"invalid source action step {source_action_step}")
        rgb = rgb_source / f"{source_frame_step}.png"
        depth_png = depth_source / f"{source_frame_step}.png"
        depth_npy = depth_source / f"{source_frame_step}.npy"
        for required in (rgb, depth_png, depth_npy):
            if not required.is_file():
                raise ValueError(f"missing aligned frame {required}")
        link_or_copy(rgb, rgb_dest / f"{output_step}.png")
        link_or_copy(depth_png, depth_dest / f"{output_step}.png")
        link_or_copy(depth_npy, depth_dest / f"{output_step}.npy")
        exported = dict(row)
        exported["step"] = output_step
        exported["source_action_step"] = source_action_step
        exported["source_frame_step"] = source_frame_step
        output_rows.append(exported)
        action_counts[action] += 1

    csv_path = destination / "keyboard_actions.csv"
    fields = list(output_rows[0])
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows), action_counts


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {args.output_root}; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)

    exported_manifest = []
    summary = {
        "manifest": str(args.manifest.resolve()),
        "raw_root": str(args.raw_root.resolve()),
        "episodes_considered": 0,
        "episodes_exported": 0,
        "steps_exported": 0,
        "actions": Counter(),
        "skipped": [],
    }
    for task in read_jsonl(args.manifest):
        summary["episodes_considered"] += 1
        raw_dir = args.raw_root / task["scene_name"] / task["episode_id"]
        result_rows = read_rows(raw_dir / "results.csv")
        if not result_rows:
            summary["skipped"].append({"episode": str(raw_dir), "reason": "not_finished"})
            continue
        result = result_rows[-1]
        if not args.allow_failed and not is_success(result):
            summary["skipped"].append({
                "episode": str(raw_dir), "reason": "expert_failed",
                "stop_reason": result.get("stop_reason"),
                "distance_world": result.get("distance_world"),
            })
            continue
        relative_dir = Path(task["scene_name"]) / task["episode_id"]
        destination = args.output_root / relative_dir
        destination.mkdir(parents=True)
        try:
            step_count, counts = export_episode(task, raw_dir, destination)
        except Exception as exc:
            shutil.rmtree(destination, ignore_errors=True)
            summary["skipped"].append({"episode": str(raw_dir), "reason": str(exc)})
            continue
        metadata = {
            **task,
            "expert": "astar-policy-teacher-v1",
            "policy_action_space": list(BC_ACTION_TO_LABEL),
            "frame_action_alignment": "aligned zero-based; source frame s-1 -> source action s",
            "steps": step_count,
            "result": result,
        }
        (destination / "episode_meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        exported_manifest.append({
            "episode_dir": str(relative_dir),
            "scene_name": task["scene_name"],
            "episode_id": task["episode_id"],
            "split": task["split"],
            "steps": step_count,
        })
        summary["episodes_exported"] += 1
        summary["steps_exported"] += step_count
        summary["actions"].update(counts)

    with (args.output_root / "dataset_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in exported_manifest:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary["actions"] = dict(summary["actions"])
    (args.output_root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"exported episodes={summary['episodes_exported']}/"
        f"{summary['episodes_considered']} steps={summary['steps_exported']} "
        f"actions={summary['actions']}"
    )


if __name__ == "__main__":
    main()
