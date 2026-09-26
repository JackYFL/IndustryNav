"""Create a scene-balanced PointGoal dataset split without copying frames.

The source episodes are linked into a new dataset root.  Each scene contributes
episodes to train, validation, and test, so all environments can be learned
while endpoint pairs remain disjoint across splits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def read_manifest(source_root: Path) -> list[dict]:
    path = source_root / "dataset_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assign_scene_balanced_splits(
    records: list[dict],
    *,
    train_per_scene: int | None = None,
    val_per_scene: int,
    test_per_scene: int | None = None,
    seed: int,
) -> list[dict]:
    if (train_per_scene is None) == (test_per_scene is None):
        raise ValueError(
            "Specify exactly one of train_per_scene or test_per_scene"
        )
    if train_per_scene is not None and train_per_scene <= 0:
        raise ValueError("train_per_scene must be positive")
    if val_per_scene <= 0:
        raise ValueError("val_per_scene must be positive")
    if test_per_scene is not None and test_per_scene <= 0:
        raise ValueError("test_per_scene must be positive")

    by_scene: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, record in enumerate(records):
        scene = str(record.get("scene_name", ""))
        if not scene:
            raise ValueError(f"Record {index} has no scene_name")
        by_scene[scene].append((index, record))

    assigned: list[dict | None] = [None] * len(records)
    for scene_index, scene in enumerate(sorted(by_scene)):
        scene_records = list(by_scene[scene])
        if train_per_scene is None:
            assert test_per_scene is not None
            required = val_per_scene + test_per_scene + 1
            scene_train_count = len(scene_records) - val_per_scene - test_per_scene
        else:
            required = train_per_scene + val_per_scene + 1
            scene_train_count = train_per_scene
        if len(scene_records) < required:
            raise ValueError(
                f"{scene}: need at least {required} episodes for train/val/test, "
                f"found {len(scene_records)}"
            )
        random.Random(seed + 1009 * scene_index).shuffle(scene_records)
        for position, (original_index, record) in enumerate(scene_records):
            if position < scene_train_count:
                split = "train"
            elif position < scene_train_count + val_per_scene:
                split = "val"
            else:
                split = "test"
            assigned[original_index] = {
                **record,
                "source_split": record.get("split"),
                "split": split,
            }
    return [record for record in assigned if record is not None]


def materialize_linked_dataset(
    source_root: Path,
    output_root: Path,
    records: list[dict],
    *,
    overwrite: bool,
) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root:
        raise ValueError("Output root must differ from source root")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    counts = Counter()
    scene_splits: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        relative = Path(str(record["episode_dir"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe episode_dir: {relative}")
        source = (source_root / relative).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Episode directory not found: {source}")
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Duplicate episode_dir: {relative}")
        destination.symlink_to(
            os.path.relpath(source, destination.parent), target_is_directory=True
        )
        split = str(record["split"])
        counts[split] += 1
        scene_splits[str(record["scene_name"])][split] += 1

    with (output_root / "dataset_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "source_dataset": str(source_root),
        "episodes": len(records),
        "scenes": len(scene_splits),
        "splits": dict(counts),
        "scene_splits": {
            scene: dict(scene_splits[scene]) for scene in sorted(scene_splits)
        },
        "storage": "relative episode-directory symlinks",
    }
    (output_root / "resplit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument(
        "--train-per-scene",
        type=int,
        help="Fixed training count; all remaining episodes after validation become test.",
    )
    split_group.add_argument(
        "--test-per-scene",
        type=int,
        help="Fixed test count; all remaining episodes after validation become train.",
    )
    parser.add_argument("--val-per-scene", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--expected-scenes", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_manifest(args.source_root)
    assigned = assign_scene_balanced_splits(
        records,
        train_per_scene=args.train_per_scene,
        val_per_scene=args.val_per_scene,
        test_per_scene=args.test_per_scene,
        seed=args.seed,
    )
    scenes = {str(record["scene_name"]) for record in assigned}
    if args.expected_scenes and len(scenes) != args.expected_scenes:
        raise SystemExit(
            f"Expected {args.expected_scenes} scenes, found {len(scenes)}"
        )
    summary = materialize_linked_dataset(
        args.source_root, args.output_root, assigned, overwrite=args.overwrite
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
