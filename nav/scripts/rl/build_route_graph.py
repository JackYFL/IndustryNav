"""Build a compact scene route graph from successful A* trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from nav.config import SCENE_ID_MAP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size-m", type=float, default=0.5)
    parser.add_argument("--lookahead-m", type=float, default=8.0)
    parser.add_argument("--max-snap-distance-m", type=float, default=4.0)
    return parser.parse_args()


def _episode_csv(record: dict, manifest_path: Path) -> Path:
    episode_dir = Path(str(record["episode_dir"]))
    if episode_dir.is_absolute():
        root = episode_dir
    elif record.get("source_dataset"):
        root = Path(str(record["source_dataset"])) / episode_dir
    else:
        root = manifest_path.parent / episode_dir
    return root / "keyboard_actions.csv"


def _positions(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    positions: list[tuple[float, float]] = []
    for row in rows:
        try:
            point = (float(row["curr_world_x"]), float(row["curr_world_z"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not positions or math.dist(point, positions[-1]) >= 0.05:
            positions.append(point)
    return positions


def build_route_graph(
    manifests: list[Path],
    *,
    grid_size_m: float,
    lookahead_m: float,
    max_snap_distance_m: float,
) -> dict:
    if min(grid_size_m, lookahead_m, max_snap_distance_m) <= 0.0:
        raise ValueError("route graph distances must be positive")
    accumulators: dict[int, dict[tuple[int, int], list[float]]] = defaultdict(dict)
    scene_edges: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(set)
    seen_episodes: set[tuple[str, str, str]] = set()
    missing = 0
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            scene_name = str(record["scene_name"])
            if scene_name not in SCENE_ID_MAP:
                continue
            path = _episode_csv(record, manifest)
            episode_key = (
                scene_name,
                str(record.get("episode_id", record.get("episode_dir"))),
                str(path),
            )
            if episode_key in seen_episodes:
                continue
            seen_episodes.add(episode_key)
            if not path.is_file():
                missing += 1
                continue
            positions = _positions(path)
            if len(positions) < 2:
                continue
            scene_id = SCENE_ID_MAP[scene_name]
            route_cells: list[tuple[int, int]] = []
            for x, z in positions:
                cell = (round(x / grid_size_m), round(z / grid_size_m))
                aggregate = accumulators[scene_id].setdefault(
                    cell, [0.0, 0.0, 0.0]
                )
                aggregate[0] += x
                aggregate[1] += z
                aggregate[2] += 1.0
                if not route_cells or cell != route_cells[-1]:
                    route_cells.append(cell)
            for first, second in zip(route_cells, route_cells[1:]):
                scene_edges[scene_id].add((first, second))

    scenes = {}
    for scene_id, cells in accumulators.items():
        ordered_cells = sorted(cells)
        indices = {cell: index for index, cell in enumerate(ordered_cells)}
        nodes = [
            [cells[cell][0] / cells[cell][2], cells[cell][1] / cells[cell][2]]
            for cell in ordered_cells
        ]
        edges = sorted({
            tuple(sorted((indices[first], indices[second])))
            for first, second in scene_edges[scene_id]
            if first != second
        })
        scenes[str(scene_id)] = {"nodes": nodes, "edges": edges}
    return {
        "format_version": 1,
        "config": {
            "grid_size_m": grid_size_m,
            "lookahead_m": lookahead_m,
            "max_snap_distance_m": max_snap_distance_m,
        },
        "stats": {
            "episodes": len(seen_episodes),
            "missing": missing,
            "scenes": len(scenes),
            "nodes": sum(len(scene["nodes"]) for scene in scenes.values()),
            "edges": sum(len(scene["edges"]) for scene in scenes.values()),
        },
        "scenes": scenes,
    }


def main() -> None:
    args = parse_args()
    payload = build_route_graph(
        args.dataset_manifest,
        grid_size_m=args.grid_size_m,
        lookahead_m=args.lookahead_m,
        max_snap_distance_m=args.max_snap_distance_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
