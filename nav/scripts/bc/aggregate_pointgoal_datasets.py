"""Create a lightweight aggregate dataset from expert and DAgger episodes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(root: Path) -> list[dict]:
    path = root / "dataset_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    roots = [root.resolve() for root in args.dataset_root]
    if output in roots:
        raise SystemExit("--output-root must differ from every source dataset")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    records = []
    seen = {}
    counts = Counter()
    for root in roots:
        for record in read_manifest(root):
            relative = Path(str(record["episode_dir"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe episode_dir: {relative}")
            source = (root / relative).resolve()
            if not source.is_dir():
                raise FileNotFoundError(f"Episode directory not found: {source}")
            key = str(relative)
            if key in seen:
                raise ValueError(
                    f"Duplicate episode_dir {key!r} from {root} and {seen[key]}"
                )
            seen[key] = root
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(
                os.path.relpath(source, destination.parent),
                target_is_directory=True,
            )
            records.append({
                **record,
                "episode_dir": key,
                "source_dataset": str(root),
            })
            counts[str(record["split"])] += 1

    with (output / "dataset_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "source_datasets": [str(root) for root in roots],
        "episodes": len(records),
        "splits": dict(counts),
        "storage": "relative episode-directory symlinks",
    }
    (output / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
