"""Validate an exported point-goal dataset before BC training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from nav.config import BC_ACTION_TO_LABEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--min-train-episodes", type=int, default=0)
    parser.add_argument("--min-val-episodes", type=int, default=0)
    parser.add_argument("--min-test-episodes", type=int, default=0)
    parser.add_argument("--require-all-actions", action="store_true")
    args = parser.parse_args()
    manifest = args.data_root / "dataset_manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"Missing {manifest}")
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    errors = []
    splits = Counter()
    actions = Counter()
    steps = 0
    for record in records:
        episode = args.data_root / record["episode_dir"]
        splits[record["split"]] += 1
        csv_path = episode / "keyboard_actions.csv"
        if not csv_path.is_file():
            errors.append(f"{episode}: missing keyboard_actions.csv")
            continue
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for expected_step, row in enumerate(rows):
            if int(float(row["step"])) != expected_step:
                errors.append(f"{episode}: non-contiguous step at {expected_step}")
            action = str(row.get("action", "")).strip().lower()
            if action not in BC_ACTION_TO_LABEL:
                errors.append(f"{episode}: invalid action {action!r}")
            actions[action] += 1
            for path in (
                episode / "keyboard_fp" / f"{expected_step}.png",
                episode / "keyboard_depth" / f"{expected_step}.png",
                episode / "keyboard_depth" / f"{expected_step}.npy",
            ):
                if not path.is_file():
                    errors.append(f"{episode}: missing {path.name}")
            steps += 1
    report = {
        "episodes": len(records), "steps": steps, "splits": dict(splits),
        "actions": dict(actions), "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(f"Dataset validation failed with {len(errors)} error(s)")
    if not actions.get("stop"):
        raise SystemExit("Dataset contains no stop targets")
    minimums = {
        "train": args.min_train_episodes,
        "val": args.min_val_episodes,
        "test": args.min_test_episodes,
    }
    for split, minimum in minimums.items():
        if splits[split] < minimum:
            raise SystemExit(
                f"Dataset has {splits[split]} {split} episodes; need at least {minimum}"
            )
    if args.require_all_actions:
        missing_actions = sorted(set(BC_ACTION_TO_LABEL) - set(actions))
        if missing_actions:
            raise SystemExit(f"Dataset is missing actions: {missing_actions}")


if __name__ == "__main__":
    main()
