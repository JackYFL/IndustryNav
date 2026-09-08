"""Build a deterministic point-goal A* pilot manifest.

The four canonical benchmark pairs are kept out of training.  We reuse their
known-safe starts and targets in cross-pair combinations, which gives five new
start/goal tasks per scene without guessing navigability from raw pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nav.config import SCENE_CODES, SCENE_ID_MAP


PAIR_INDICES = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))


def split_for_scene(scene: str) -> str:
    scene_number = int(scene.removeprefix("scene"))
    if scene_number <= 16:
        return "train"
    if scene_number <= 20:
        return "val"
    return "test"


def build_manifest(
    points: dict,
    seeds: int = 2,
    pairing: str = "cross",
) -> list[dict]:
    if pairing not in {"cross", "canonical", "both"}:
        raise ValueError("pairing must be 'cross', 'canonical', or 'both'")
    pair_groups = []
    if pairing in {"canonical", "both"}:
        pair_groups.append(("canonical", tuple((i, i) for i in range(4))))
    if pairing in {"cross", "both"}:
        pair_groups.append(("pg", PAIR_INDICES))
    records = []
    for scene in SCENE_CODES:
        entries = points.get(scene, [])
        if len(entries) < 4:
            raise ValueError(f"{scene} needs at least four canonical point pairs")
        for seed in range(seeds):
            for prefix, pairs in pair_groups:
                for pair_index, (start_index, target_index) in enumerate(pairs, 1):
                    source_start = entries[start_index]
                    source_target = entries[target_index]
                    start = source_start["start"]
                    target = source_target["target"]
                    episode_id = f"{prefix}{pair_index:02d}_seed{seed}"
                    records.append(
                        {
                            "episode_id": episode_id,
                            "scene_name": scene,
                            "scene_id": SCENE_ID_MAP[scene],
                            "split": split_for_scene(scene),
                            "seed": seed,
                            "pair_index": pair_index,
                            "pairing": "cross" if prefix == "pg" else "canonical",
                            "source_start_point": str(source_start["point_id"]),
                            "source_target_point": str(source_target["point_id"]),
                            "init_world_x": float(start["x"]),
                            "init_world_z": float(start["z"]),
                            "init_direction": (
                                float(start.get("direction", 180.0)) + 45.0 * seed
                            )
                            % 360.0,
                            "target_x": int(target["x"]),
                            "target_y": int(target["y"]),
                            # Deterministic near-80/20 moving/static mixture.
                            "dynamic_objects": (
                                "static" if (pair_index + seed) % 5 == 0 else "moving"
                            ),
                            "motion_random_seed": 1000 + seed,
                            "light_random_seed": 2000 + seed,
                        }
                    )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("input_points.json"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("datasets/astar_pointgoal_pilot/collection_manifest.jsonl"),
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument(
        "--pairing", choices=("cross", "canonical", "both"), default="cross"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    points = json.loads(args.input.read_text(encoding="utf-8"))
    records = build_manifest(points, args.seeds, pairing=args.pairing)
    write_jsonl(args.output, records)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    spec = {
        "name": f"astar_pointgoal_pilot_{args.pairing}_v1",
        "episodes": len(records),
        "scenes": len(SCENE_CODES),
        "pairs_per_scene": len(records) // (len(SCENE_CODES) * args.seeds),
        "seeds": args.seeds,
        "split_policy": "scene1-16 train, scene17-20 val, scene21-24 test",
        "pairing": args.pairing,
        "canonical_benchmark_pairs_included": args.pairing in {"canonical", "both"},
        "action_space": ["forward", "stop", "turn right", "turn left"],
        "action_controls": {
            "forward_signal": 7.5,
            "observed_forward_step_m": 0.75,
            "turn_right_signal": 11.25,
            "turn_left_signal": -11.25,
            "observed_turn_degrees": 22.5,
        },
        "sim_steps_per_decision": 1,
        "runtime_lighting": "scene-authored fixed lighting/exposure",
        "lighting_augmentation": "apply photometric augmentation during policy training",
        "frame_action_alignment": "source frame s-1 supervises source action step s",
        "manifest_sha256": digest,
    }
    spec_path = args.output.with_name("dataset_spec.json")
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    counts = {split: sum(r["split"] == split for r in records) for split in ("train", "val", "test")}
    print(f"Wrote {len(records)} episodes to {args.output} | splits={counts} | sha256={digest}")


if __name__ == "__main__":
    main()
