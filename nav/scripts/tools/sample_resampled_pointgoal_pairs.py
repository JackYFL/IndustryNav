"""Sample new PointGoal tasks from interior states of successful A* routes.

The benchmark tasks in ``input_points.json`` remain evaluation-only.  This
sampler uses positions that an A* agent actually traversed, but rejects route
endpoints and every candidate close to a benchmark endpoint.  Start and target
coordinates in the resulting manifest are therefore new while retaining strong
evidence that both endpoints lie on the same connected free-space route.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nav.config import SCENE_CODES, SCENE_ID_MAP, UNITY_MAP_SIZE


SUCCESS_REASONS = {"astar_stop", "reached_vicinity"}


@dataclass(frozen=True)
class RoutePoint:
    step: int
    world_x: float
    world_z: float
    pixel_x: int
    pixel_y: int
    cumulative_m: float


@dataclass(frozen=True)
class Route:
    scene_name: str
    episode_id: str
    points: tuple[RoutePoint, ...]


@dataclass(frozen=True)
class PairCandidate:
    route: Route
    start: RoutePoint
    target: RoutePoint
    route_distance_m: float
    euclidean_distance_m: float


def split_for_scene(scene_name: str) -> str:
    scene_number = int(scene_name.removeprefix("scene"))
    if scene_number <= 16:
        return "train"
    if scene_number <= 20:
        return "val"
    return "test"


def _as_float(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}={row[key]!r}")
    return value


def _successful_metadata(metadata: dict) -> bool:
    result = metadata.get("result", {})
    try:
        distance_m = float(result.get("distance_world", "inf"))
    except (TypeError, ValueError):
        return False
    return result.get("stop_reason") in SUCCESS_REASONS and distance_m <= 2.0


def _read_route(episode_dir: Path) -> Route | None:
    metadata_path = episode_dir / "episode_meta.json"
    actions_path = episode_dir / "keyboard_actions.csv"
    if not metadata_path.is_file() or not actions_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not _successful_metadata(metadata):
        return None

    raw_points: list[tuple[int, float, float, int, int]] = []
    try:
        with actions_path.open(newline="") as stream:
            for fallback_step, row in enumerate(csv.DictReader(stream)):
                world_x = _as_float(row, "curr_world_x")
                world_z = _as_float(row, "curr_world_z")
                pixel_x = int(round(_as_float(row, "curr_px")))
                pixel_y = int(round(_as_float(row, "curr_py")))
                step = int(float(row.get("step", fallback_step)))
                if not (0 <= pixel_x < int(UNITY_MAP_SIZE[0])):
                    continue
                if not (0 <= pixel_y < int(UNITY_MAP_SIZE[1])):
                    continue
                if raw_points and math.hypot(
                    world_x - raw_points[-1][1], world_z - raw_points[-1][2]
                ) < 0.05:
                    continue
                raw_points.append((step, world_x, world_z, pixel_x, pixel_y))
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if len(raw_points) < 3:
        return None

    points: list[RoutePoint] = []
    cumulative_m = 0.0
    previous: tuple[int, float, float, int, int] | None = None
    for point in raw_points:
        if previous is not None:
            segment_m = math.hypot(point[1] - previous[1], point[2] - previous[2])
            # A discontinuity indicates a reset/teleport rather than traversed free space.
            if segment_m > 3.0:
                return None
            cumulative_m += segment_m
        points.append(
            RoutePoint(
                step=point[0],
                world_x=point[1],
                world_z=point[2],
                pixel_x=point[3],
                pixel_y=point[4],
                cumulative_m=cumulative_m,
            )
        )
        previous = point
    return Route(
        scene_name=str(metadata["scene_name"]),
        episode_id=str(metadata["episode_id"]),
        points=tuple(points),
    )


def load_routes(dataset_root: Path) -> dict[str, list[Route]]:
    routes = {scene: [] for scene in SCENE_CODES}
    for metadata_path in sorted(dataset_root.glob("scene*/*/episode_meta.json")):
        route = _read_route(metadata_path.parent)
        if route is not None and route.scene_name in routes:
            routes[route.scene_name].append(route)
    return routes


def load_benchmark_points(path: Path) -> dict[str, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {scene: list(payload.get(scene, [])) for scene in SCENE_CODES}


def reference_world_endpoints(routes: Iterable[Route]) -> tuple[tuple[float, float], ...]:
    """Return endpoints of source routes, which correspond to old benchmark points."""
    endpoints: list[tuple[float, float]] = []
    for route in routes:
        endpoints.append((route.points[0].world_x, route.points[0].world_z))
        endpoints.append((route.points[-1].world_x, route.points[-1].world_z))
    return tuple(endpoints)


def _far_from_world_endpoints(
    point: RoutePoint,
    endpoints: Iterable[tuple[float, float]],
    clearance_m: float,
) -> bool:
    return all(
        math.hypot(point.world_x - world_x, point.world_z - world_z) >= clearance_m
        for world_x, world_z in endpoints
    )


def _far_from_benchmark_targets(
    point: RoutePoint,
    benchmark_entries: Iterable[dict],
    clearance_px: float,
) -> bool:
    for entry in benchmark_entries:
        target = entry["target"]
        if math.hypot(point.pixel_x - float(target["x"]), point.pixel_y - float(target["y"])) < clearance_px:
            return False
    return True


def _far_from_pixel_endpoints(
    point: RoutePoint,
    endpoints: Iterable[tuple[float, float]],
    clearance_px: float,
) -> bool:
    return all(
        math.hypot(point.pixel_x - pixel_x, point.pixel_y - pixel_y) >= clearance_px
        for pixel_x, pixel_y in endpoints
    )


def load_manifest_exclusions(
    paths: Iterable[Path],
) -> tuple[
    dict[str, list[tuple[float, float]]],
    dict[str, list[tuple[float, float]]],
]:
    """Load both endpoints of prior manifests so a new sample cannot reuse them."""
    world = {scene: [] for scene in SCENE_CODES}
    pixels = {scene: [] for scene in SCENE_CODES}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                task = json.loads(line)
                scene = str(task["scene_name"])
                if scene not in world:
                    continue
                world[scene].append(
                    (float(task["init_world_x"]), float(task["init_world_z"]))
                )
                if "sampled_target_world_x" in task and "sampled_target_world_z" in task:
                    world[scene].append(
                        (
                            float(task["sampled_target_world_x"]),
                            float(task["sampled_target_world_z"]),
                        )
                    )
                pixels[scene].append((float(task["target_x"]), float(task["target_y"])))
    return world, pixels


def candidate_pairs_for_scene(
    routes: list[Route],
    benchmark_entries: list[dict],
    *,
    min_route_distance_m: float,
    max_route_distance_m: float,
    min_euclidean_distance_m: float,
    benchmark_clearance_m: float,
    benchmark_target_clearance_px: float,
    route_endpoint_clearance_m: float,
    candidate_spacing_m: float,
    excluded_world_endpoints: Iterable[tuple[float, float]] = (),
    excluded_target_pixels: Iterable[tuple[float, float]] = (),
    excluded_world_clearance_m: float = 0.0,
    excluded_target_clearance_px: float = 0.0,
) -> list[PairCandidate]:
    benchmark_world_starts = tuple(
        (float(entry["start"]["x"]), float(entry["start"]["z"]))
        for entry in benchmark_entries
    )
    excluded_world_endpoints = tuple(excluded_world_endpoints)
    excluded_target_pixels = tuple(excluded_target_pixels)
    candidates: list[PairCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    for route in routes:
        total_m = route.points[-1].cumulative_m
        eligible: list[RoutePoint] = []
        last_kept_m = -math.inf
        for point in route.points:
            if point.cumulative_m < route_endpoint_clearance_m:
                continue
            if total_m - point.cumulative_m < route_endpoint_clearance_m:
                continue
            if point.cumulative_m - last_kept_m < candidate_spacing_m:
                continue
            if not _far_from_world_endpoints(
                point, benchmark_world_starts, benchmark_clearance_m
            ):
                continue
            if not _far_from_benchmark_targets(
                point, benchmark_entries, benchmark_target_clearance_px
            ):
                continue
            if not _far_from_world_endpoints(
                point, excluded_world_endpoints, excluded_world_clearance_m
            ):
                continue
            if not _far_from_pixel_endpoints(
                point, excluded_target_pixels, excluded_target_clearance_px
            ):
                continue
            eligible.append(point)
            last_kept_m = point.cumulative_m

        for left_index, left in enumerate(eligible):
            for right in eligible[left_index + 1 :]:
                route_distance_m = right.cumulative_m - left.cumulative_m
                if route_distance_m < min_route_distance_m:
                    continue
                if route_distance_m > max_route_distance_m:
                    break
                euclidean_m = math.hypot(
                    right.world_x - left.world_x, right.world_z - left.world_z
                )
                if euclidean_m < min_euclidean_distance_m:
                    continue
                for start, target in ((left, right), (right, left)):
                    key = (
                        round(start.world_x * 2), round(start.world_z * 2),
                        round(target.world_x * 2), round(target.world_z * 2),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        PairCandidate(
                            route=route,
                            start=start,
                            target=target,
                            route_distance_m=route_distance_m,
                            euclidean_distance_m=euclidean_m,
                        )
                    )
    return candidates


def _candidate_is_separated(
    candidate: PairCandidate,
    selected: list[PairCandidate],
    separation_m: float,
) -> bool:
    if separation_m <= 0:
        return True
    for existing in selected:
        start_distance = math.hypot(
            candidate.start.world_x - existing.start.world_x,
            candidate.start.world_z - existing.start.world_z,
        )
        target_distance = math.hypot(
            candidate.target.world_x - existing.target.world_x,
            candidate.target.world_z - existing.target.world_z,
        )
        if start_distance < separation_m or target_distance < separation_m:
            return False
    return True


def select_pairs(
    candidates: list[PairCandidate],
    count: int,
    *,
    seed: int,
    endpoint_separation_m: float,
) -> list[PairCandidate]:
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    # Interleave short, medium, and long routes instead of allowing the much
    # larger short-route candidate pool to dominate the manifest.
    pools = [
        [item for item in shuffled if item.route_distance_m < 25.0],
        [item for item in shuffled if 25.0 <= item.route_distance_m < 40.0],
        [item for item in shuffled if item.route_distance_m >= 40.0],
    ]
    ordered: list[PairCandidate] = []
    while any(pools):
        for pool in pools:
            if pool:
                ordered.append(pool.pop())

    selected: list[PairCandidate] = []
    source_counts: dict[str, int] = {}
    max_per_source = max(2, math.ceil(count / 4))
    selection_stages = (
        (endpoint_separation_m, max_per_source),
        (endpoint_separation_m / 2.0, max_per_source * 2),
        (0.0, count),
    )
    for separation, source_limit in selection_stages:
        for candidate in ordered:
            if candidate in selected:
                continue
            source = candidate.route.episode_id
            if source_counts.get(source, 0) >= source_limit:
                continue
            if not _candidate_is_separated(candidate, selected, separation):
                continue
            selected.append(candidate)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(selected) == count:
                return selected
    return selected


def build_manifest(
    routes_by_scene: dict[str, list[Route]],
    benchmark_points: dict[str, list[dict]],
    *,
    pairs_per_scene: int,
    seed: int,
    min_route_distance_m: float,
    max_route_distance_m: float,
    min_euclidean_distance_m: float,
    benchmark_clearance_m: float,
    benchmark_target_clearance_px: float,
    route_endpoint_clearance_m: float,
    candidate_spacing_m: float,
    endpoint_separation_m: float,
    excluded_world_endpoints: dict[str, list[tuple[float, float]]] | None = None,
    excluded_target_pixels: dict[str, list[tuple[float, float]]] | None = None,
    excluded_world_clearance_m: float = 0.0,
    excluded_target_clearance_px: float = 0.0,
) -> list[dict]:
    excluded_world_endpoints = excluded_world_endpoints or {}
    excluded_target_pixels = excluded_target_pixels or {}
    manifest: list[dict] = []
    for scene_name in SCENE_CODES:
        candidates = candidate_pairs_for_scene(
            routes_by_scene.get(scene_name, []),
            benchmark_points.get(scene_name, []),
            min_route_distance_m=min_route_distance_m,
            max_route_distance_m=max_route_distance_m,
            min_euclidean_distance_m=min_euclidean_distance_m,
            benchmark_clearance_m=benchmark_clearance_m,
            benchmark_target_clearance_px=benchmark_target_clearance_px,
            route_endpoint_clearance_m=route_endpoint_clearance_m,
            candidate_spacing_m=candidate_spacing_m,
            excluded_world_endpoints=excluded_world_endpoints.get(scene_name, []),
            excluded_target_pixels=excluded_target_pixels.get(scene_name, []),
            excluded_world_clearance_m=excluded_world_clearance_m,
            excluded_target_clearance_px=excluded_target_clearance_px,
        )
        scene_seed = seed + 1009 * SCENE_ID_MAP[scene_name]
        selected = select_pairs(
            candidates,
            pairs_per_scene,
            seed=scene_seed,
            endpoint_separation_m=endpoint_separation_m,
        )
        if len(selected) != pairs_per_scene:
            raise ValueError(
                f"{scene_name}: requested {pairs_per_scene} pairs but only "
                f"selected {len(selected)} from {len(candidates)} candidates"
            )
        rng = random.Random(scene_seed)
        for pair_index, candidate in enumerate(selected, 1):
            episode_id = f"resampled{pair_index:02d}"
            motion_seed = 50000 + SCENE_ID_MAP[scene_name] * 100 + pair_index
            manifest.append(
                {
                    "episode_id": episode_id,
                    "scene_name": scene_name,
                    "scene_id": SCENE_ID_MAP[scene_name],
                    "split": split_for_scene(scene_name),
                    "seed": motion_seed,
                    "pair_index": pair_index,
                    "pairing": "resampled_route_interior",
                    "init_world_x": round(candidate.start.world_x, 6),
                    "init_world_z": round(candidate.start.world_z, 6),
                    "init_direction": float(rng.randrange(8) * 45),
                    "target_x": candidate.target.pixel_x,
                    "target_y": candidate.target.pixel_y,
                    "sampled_target_world_x": round(candidate.target.world_x, 6),
                    "sampled_target_world_z": round(candidate.target.world_z, 6),
                    "source_episode": candidate.route.episode_id,
                    "source_start_step": candidate.start.step,
                    "source_target_step": candidate.target.step,
                    "source_route_distance_m": round(candidate.route_distance_m, 4),
                    "euclidean_distance_m": round(candidate.euclidean_distance_m, 4),
                    "dynamic_objects": "static" if pair_index % 5 == 0 else "moving",
                    "motion_random_seed": motion_seed,
                    "light_random_seed": 60000 + SCENE_ID_MAP[scene_name] * 100 + pair_index,
                }
            )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-root", type=Path,
        default=Path("datasets/astar_pointgoal_expanded_v3/bc"),
    )
    parser.add_argument("--benchmark-points", type=Path, default=Path("input_points.json"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("datasets/astar_pointgoal_resampled_v1/collection_manifest.jsonl"),
    )
    parser.add_argument("--pairs-per-scene", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--min-route-distance-m", type=float, default=12.0)
    parser.add_argument("--max-route-distance-m", type=float, default=65.0)
    parser.add_argument("--min-euclidean-distance-m", type=float, default=8.0)
    parser.add_argument("--benchmark-clearance-m", type=float, default=3.0)
    parser.add_argument("--benchmark-target-clearance-px", type=float, default=20.0)
    parser.add_argument("--route-endpoint-clearance-m", type=float, default=4.0)
    parser.add_argument("--candidate-spacing-m", type=float, default=2.0)
    parser.add_argument("--endpoint-separation-m", type=float, default=2.5)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Prior manifest whose start and target endpoints must not be reused.",
    )
    parser.add_argument("--excluded-world-clearance-m", type=float, default=2.0)
    parser.add_argument("--excluded-target-clearance-px", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs_per_scene <= 0:
        raise SystemExit("--pairs-per-scene must be positive")
    routes = load_routes(args.trajectory_root)
    benchmark_points = load_benchmark_points(args.benchmark_points)
    excluded_world, excluded_pixels = load_manifest_exclusions(args.exclude_manifest)
    manifest = build_manifest(
        routes,
        benchmark_points,
        pairs_per_scene=args.pairs_per_scene,
        seed=args.seed,
        min_route_distance_m=args.min_route_distance_m,
        max_route_distance_m=args.max_route_distance_m,
        min_euclidean_distance_m=args.min_euclidean_distance_m,
        benchmark_clearance_m=args.benchmark_clearance_m,
        benchmark_target_clearance_px=args.benchmark_target_clearance_px,
        route_endpoint_clearance_m=args.route_endpoint_clearance_m,
        candidate_spacing_m=args.candidate_spacing_m,
        endpoint_separation_m=args.endpoint_separation_m,
        excluded_world_endpoints=excluded_world,
        excluded_target_pixels=excluded_pixels,
        excluded_world_clearance_m=args.excluded_world_clearance_m,
        excluded_target_clearance_px=args.excluded_target_clearance_px,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for task in manifest:
            stream.write(json.dumps(task, sort_keys=True) + "\n")
    manifest_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    split_counts = {
        split: sum(task["split"] == split for task in manifest)
        for split in ("train", "val", "test")
    }
    spec = {
        "name": "astar_pointgoal_resampled_v1",
        "sampling": "interior states of successful A* trajectories",
        "benchmark_points_role": "exclusion-only; never used as sampled endpoints",
        "benchmark_points_sha256": hashlib.sha256(args.benchmark_points.read_bytes()).hexdigest(),
        "source_trajectory_root": str(args.trajectory_root.resolve()),
        "source_successful_routes": sum(len(value) for value in routes.values()),
        "scenes": len(SCENE_CODES),
        "pairs_per_scene": args.pairs_per_scene,
        "episodes": len(manifest),
        "split_counts": split_counts,
        "split_policy": "scene1-16 train, scene17-20 val, scene21-24 test",
        "benchmark_clearance_m": args.benchmark_clearance_m,
        "benchmark_target_clearance_px": args.benchmark_target_clearance_px,
        "excluded_manifests": [str(path.resolve()) for path in args.exclude_manifest],
        "excluded_manifest_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.exclude_manifest
        },
        "excluded_world_clearance_m": args.excluded_world_clearance_m,
        "excluded_target_clearance_px": args.excluded_target_clearance_px,
        "route_endpoint_clearance_m": args.route_endpoint_clearance_m,
        "route_distance_range_m": [args.min_route_distance_m, args.max_route_distance_m],
        "min_euclidean_distance_m": args.min_euclidean_distance_m,
        "seed": args.seed,
        "manifest_sha256": manifest_sha256,
    }
    args.output.with_name("dataset_spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(manifest)} new pairs to {args.output} | "
        f"splits={split_counts} | source_routes={spec['source_successful_routes']} | "
        f"sha256={manifest_sha256}"
    )


if __name__ == "__main__":
    main()
