"""Export DAgger rollouts as policy observations labeled by the A* oracle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from nav.config import ACTION_SPACE_POINTGOAL, BC_NAV_ACTION_TO_LABEL
from nav.scripts.astar.collect_astar_pointgoal import read_jsonl
from nav.scripts.astar.export_astar_pointgoal_dataset import link_or_copy, read_rows
from nav.utils import action2signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def export_episode(raw_dir: Path, destination: Path) -> tuple[int, Counter]:
    action_rows = {
        int(float(row["step"])): row
        for row in read_rows(raw_dir / "dagger_actions.csv")
    }
    labels = read_rows(raw_dir / "dagger_labels.csv")
    if not action_rows or not labels:
        raise ValueError("missing DAgger actions or oracle labels")

    rgb_source = raw_dir / "dagger_fp"
    depth_source = raw_dir / "dagger_depth"
    rgb_dest = destination / "keyboard_fp"
    depth_dest = destination / "keyboard_depth"
    rgb_dest.mkdir(parents=True)
    depth_dest.mkdir(parents=True)
    output_rows = []
    counts = Counter()
    for label in labels:
        expert_action = str(label.get("expert_action", "")).strip().lower()
        if expert_action not in BC_NAV_ACTION_TO_LABEL:
            continue
        action_step = int(float(label["step"]))
        frame_step = int(float(label["frame_step"]))
        row = action_rows.get(action_step)
        if row is None:
            raise ValueError(f"missing behavior action row {action_step}")
        rgb = rgb_source / f"{frame_step}.png"
        depth_png = depth_source / f"{frame_step}.png"
        depth_npy = depth_source / f"{frame_step}.npy"
        for required in (rgb, depth_png, depth_npy):
            if not required.is_file():
                raise ValueError(f"missing aligned frame {required}")

        output_step = len(output_rows)
        link_or_copy(rgb, rgb_dest / f"{output_step}.png")
        link_or_copy(depth_png, depth_dest / f"{output_step}.png")
        link_or_copy(depth_npy, depth_dest / f"{output_step}.npy")
        signal = action2signal(expert_action, ACTION_SPACE_POINTGOAL)[0]
        exported = dict(row)
        exported.update({
            "step": output_step,
            "action": expert_action,
            "move": float(signal[0]),
            "strafe": float(signal[1]),
            "look": float(signal[2]),
            "source_action_step": action_step,
            "source_frame_step": frame_step,
            "behavior_action": label["executed_action"],
            "policy_action": label["policy_action"],
            "dagger_beta": label["beta"],
            "dagger_used_expert": label["used_expert"],
            "dagger_disagreement": label["disagreement"],
        })
        output_rows.append(exported)
        counts[expert_action] += 1
        counts["disagreement"] += str(label["disagreement"]).lower() == "true"
        counts["intervention"] += str(label["used_expert"]).lower() == "true"

    if not output_rows:
        raise ValueError("no navigation oracle labels")
    with (destination / "keyboard_actions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows), counts


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {args.output_root}; pass --overwrite")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)

    exported_manifest = []
    summary = {
        "round_id": args.round_id,
        "manifest": str(args.manifest.resolve()),
        "raw_root": str(args.raw_root.resolve()),
        "episodes_considered": 0,
        "episodes_exported": 0,
        "steps_exported": 0,
        "counts": Counter(),
        "skipped": [],
    }
    for task in read_jsonl(args.manifest):
        if task["split"] != "train":
            continue
        summary["episodes_considered"] += 1
        raw_dir = args.raw_root / task["scene_name"] / task["episode_id"]
        episode_id = f"dagger_{args.round_id}_{task['episode_id']}"
        relative = Path(task["scene_name"]) / episode_id
        destination = args.output_root / relative
        destination.mkdir(parents=True)
        try:
            steps, counts = export_episode(raw_dir, destination)
        except Exception as exc:
            shutil.rmtree(destination, ignore_errors=True)
            summary["skipped"].append({
                "episode": str(raw_dir), "reason": str(exc)
            })
            continue
        metadata = {
            **task,
            "episode_id": episode_id,
            "source_episode_id": task["episode_id"],
            "expert": "astar-policy-teacher-v1",
            "behavior": "dagger-mixture",
            "round_id": args.round_id,
            "steps": steps,
        }
        (destination / "episode_meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        exported_manifest.append({
            "episode_dir": str(relative),
            "scene_name": task["scene_name"],
            "episode_id": episode_id,
            "split": "train",
            "steps": steps,
            "source": "dagger",
            "round_id": args.round_id,
        })
        summary["episodes_exported"] += 1
        summary["steps_exported"] += steps
        summary["counts"].update(counts)

    with (args.output_root / "dataset_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in exported_manifest:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary["counts"] = dict(summary["counts"])
    (args.output_root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"exported episodes={summary['episodes_exported']} "
        f"steps={summary['steps_exported']} counts={summary['counts']}"
    )


if __name__ == "__main__":
    main()
