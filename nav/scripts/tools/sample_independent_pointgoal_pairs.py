"""Sample independent free-space point pairs, not interiors of old trajectories.

All benchmark starts AND goals are excluded in world coordinates in BOTH
roles. Validation endpoints are reserved before training endpoints are drawn.
An occupancy route is a connectivity estimate, not proof of physical success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from nav.data.navigation_map import NavigationMap
from nav.config import ACTION_SPACE_POINTGOAL, EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M
from nav.harness.coordinates import visual_to_world_coords


DIFFICULTIES = ("short", "medium", "long", "detour")


def validate_training_clearance(value, *, role="start"):
    """Optional metric range on the already inflated approximate map grid."""
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"Training {role} clearance requires minimum and maximum meters")
    lower, upper = map(float, value)
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower < upper):
        raise ValueError(f"Training {role} clearance must satisfy finite 0 <= minimum < maximum")
    return lower, upper


def validate_training_start_clearance(value):
    """Preserve the existing start-range helper API."""
    return validate_training_clearance(value, role="start")


def minimum_clearance(points: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    if not len(excluded):
        return np.full(len(points), np.inf)
    return np.linalg.norm(points[:, None, :] - excluded[None, :, :], axis=-1).min(axis=1)


def benchmark_endpoints(geometry: NavigationMap, entries: list[dict]) -> np.ndarray:
    meta = geometry.metadata
    endpoints = []
    for entry in entries:
        endpoints.append((entry["start"]["x"], entry["start"]["z"]))
        endpoints.append(visual_to_world_coords(meta["margin"],
            (entry["target"]["x"], entry["target"]["y"]), meta["projector"],
            map_size=meta["minimap_size"]))
    return np.asarray(endpoints, dtype=float)


def sample_scene(
    geometry: NavigationMap, entries: list[dict], *, seed: int, train_count: int,
    validation_count: int, benchmark_clearance_m: float = 3.0,
    split_clearance_m: float = 2.0, max_attempts: int = 30000,
    reserved_validation_records: list[dict] | None = None,
    training_min_distance_m: float = 8.0,
    training_long_fraction: float | None = None,
    training_episode_prefix: str = "independent_train",
    excluded_pair_records: list[dict] | None = None,
    training_start_clearance_m: tuple[float, float] | None = None,
    training_goal_clearance_m: tuple[float, float] | None = None,
) -> tuple[list[dict], dict]:
    if min(train_count, validation_count) <= 0:
        raise ValueError("Both independent splits need positive counts")
    if min(benchmark_clearance_m, split_clearance_m) <= 0:
        raise ValueError("Endpoint exclusion distances must be positive")
    if not 8 <= training_min_distance_m < 40:
        raise ValueError("Training minimum distance must be in [8, 40)")
    if training_long_fraction is not None and not 0 <= training_long_fraction <= 1:
        raise ValueError("Training long-distance fraction must be in [0, 1]")
    start_clearance_range = validate_training_start_clearance(training_start_clearance_m)
    goal_clearance_range = validate_training_clearance(training_goal_clearance_m, role="goal")
    rng = np.random.default_rng(seed)
    canonical = benchmark_endpoints(geometry, entries)
    cells, points = geometry.free_cells, geometry.free_world
    # Additional start/goal clearance on top of the map's obstacle inflation.
    clearance_cells = cv2.distanceTransform(geometry.walkable.astype(np.uint8), cv2.DIST_L2, 5)
    canonical_safe = minimum_clearance(points, canonical) >= benchmark_clearance_m
    eligible = canonical_safe & (clearance_cells[cells[:, 1], cells[:, 0]] >= 1.8)
    # Tiny image artifacts cannot become isolated sampled 'rooms'.
    sizes = np.bincount(geometry.components.ravel())
    eligible &= sizes[geometry.components[cells[:, 1], cells[:, 0]]] >= 100
    training_starts = None
    training_goals = None
    clearance_m = None
    if start_clearance_range is not None or goal_clearance_range is not None:
        # This is grid-center distance after the map's existing inflation, NOT
        # physical agent clearance. Validation always retains the original pool.
        cell_scale = float(np.linalg.svd(geometry.basis, compute_uv=False).min())
        clearance_m = clearance_cells[cells[:, 1], cells[:, 0]] * cell_scale
    if start_clearance_range is not None:
        lower, upper = start_clearance_range
        training_starts = (canonical_safe & (clearance_m >= lower) & (clearance_m <= upper)
                           & (sizes[geometry.components[cells[:, 1], cells[:, 0]]] >= 100))
    if goal_clearance_range is not None:
        lower, upper = goal_clearance_range
        training_goals = (canonical_safe & (clearance_m >= lower) & (clearance_m <= upper)
                          & (sizes[geometry.components[cells[:, 1], cells[:, 0]]] >= 100))
    def record_pair_key(record):
        start = tuple(np.rint(geometry.world_to_cell((record["init_world_x"], record["init_world_z"]))).astype(int))
        goal = tuple(np.rint(geometry.world_to_cell((record["sampled_target_world_x"], record["sampled_target_world_z"]))).astype(int))
        return tuple(sorted((start, goal)))

    selected_records = []
    reserved_endpoints = []
    pair_keys = {record_pair_key(r) for r in (excluded_pair_records or [])}
    scene = geometry.metadata["scene_name"]
    meta = geometry.metadata
    counts_report = {}
    splits = (("val", validation_count), ("train", train_count))
    if reserved_validation_records is not None:
        validation = [dict(r) for r in reserved_validation_records]
        if (len(validation) != validation_count
                or len({r["episode_id"] for r in validation}) != len(validation)
                or any(r["scene_name"] != scene or r["split"] != "val" for r in validation)):
            raise ValueError("Reserved validation must be the exact unique scene cohort")
        selected_records.extend(validation)
        for r in validation:
            reserved_endpoints.extend(((r["init_world_x"], r["init_world_z"]),
                                       (r["sampled_target_world_x"], r["sampled_target_world_z"])))
            pair_keys.add(record_pair_key(r))
        if minimum_clearance(np.asarray(reserved_endpoints), canonical).min() < benchmark_clearance_m:
            raise ValueError("Reserved validation violates canonical endpoint exclusion")
        counts_report["val"] = dict(Counter(r["difficulty"] for r in validation))
        splits = (("train", train_count),)
    for split, count in splits:
        active = eligible.copy()
        if reserved_endpoints:
            active &= minimum_clearance(points, np.asarray(reserved_endpoints)) >= split_clearance_m
        pool = np.flatnonzero(active)
        start_pool = pool
        if split == "train" and training_goals is not None:
            goal_active = training_goals.copy()
            if reserved_endpoints:
                goal_active &= minimum_clearance(points, np.asarray(reserved_endpoints)) >= split_clearance_m
            pool = np.flatnonzero(goal_active)
            if not len(pool):
                raise ValueError(f"No eligible {scene} training goals in requested clearance range; do not relax")
        if split == "train" and training_starts is not None:
            start_active = training_starts.copy()
            if reserved_endpoints:
                start_active &= minimum_clearance(points, np.asarray(reserved_endpoints)) >= split_clearance_m
            start_pool = np.flatnonzero(start_active)
            if not len(start_pool):
                raise ValueError(f"No eligible {scene} training starts in requested clearance range; do not relax")
        if len(pool) < 20 or not len(start_pool):
            raise ValueError(f"Too few independently reserved endpoints for {scene}/{split}")
        quotas = {kind: count // 4 + int(index < count % 4) for index, kind in enumerate(DIFFICULTIES)}
        distance_quota = split == "train" and training_long_fraction is not None
        if distance_quota:
            long_count = int(round(count * training_long_fraction))
            quotas = {"under40": count - long_count, "long": long_count}
        counts = Counter()
        attempts = 0
        while sum(counts.values()) < count and attempts < max_attempts:
            attempts += 1
            gi = int(rng.choice(pool))
            gx, gy = cells[gi]
            # Target pixels are integral in the protocol; use the exact rounded
            # pixel-to-world goal rather than claiming the grid center is exact.
            target_pixel = np.rint((cells[gi] + 0.5) * meta["grid_step_px"]).astype(int)
            goal = np.asarray(visual_to_world_coords(meta["margin"], tuple(target_pixel),
                meta["projector"], map_size=meta["minimap_size"]))
            if minimum_clearance(goal[None], canonical)[0] < benchmark_clearance_m:
                continue
            if reserved_endpoints and minimum_clearance(goal[None], np.asarray(reserved_endpoints))[0] < split_clearance_m:
                continue
            if split == "train" and goal_clearance_range is not None:
                # Keep the claimed grid-clearance cell valid after target-pixel
                # quantization. This is still a proxy, not physical clearance.
                rounded_goal_cell = np.rint(geometry.world_to_cell(goal)).astype(int)
                if not np.array_equal(rounded_goal_cell, cells[gi]):
                    continue
            field = geometry.distance_field(goal)
            # Inspect a random batch per target instead of a dense all-pairs
            # matrix. Seed and quotas make the result reproducible.
            choices = rng.choice(start_pool, size=min(len(start_pool), 128), replace=False)
            for si in choices:
                sx, sy = cells[si]
                start = points[si]
                distance = float(np.linalg.norm(start - goal))
                route_distance = float(field[sy, sx])
                minimum_distance = training_min_distance_m if split == "train" else 8.0
                if not minimum_distance <= distance <= 70.0 or not np.isfinite(route_distance) or route_distance > 100:
                    continue
                detour = route_distance / distance
                kind = ("detour" if detour >= 1.25 else "short" if distance < 20
                        else "medium" if distance < 40 else "long")
                stratum = ("under40" if distance < 40 else "long") if distance_quota else kind
                if counts[stratum] >= quotas[stratum]:
                    continue
                budget = int(np.clip(math.ceil(2.2 * distance + 60), 80, 320))
                # Necessary translation-only budget condition, not a claim
                # that turning, physics, and moving objects will fit as well.
                step_m = ACTION_SPACE_POINTGOAL["forward"] * EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M
                min_translation_steps = math.ceil(max(0.0, route_distance - 2.0) / step_m)
                if min_translation_steps > budget - 12:
                    continue
                key = tuple(sorted(((int(sx), int(sy)), (int(gx), int(gy)))))
                if key in pair_keys:
                    continue
                pair_keys.add(key)
                index = sum(counts.values())
                record = dict(scene_name=scene, scene_id=int(meta["scene_id"]), split=split,
                    episode_id=(f"{training_episode_prefix}_{index:04d}" if split == "train"
                                else f"independent_val_{index:04d}"), pairing="independent_occupancy",
                    init_world_x=float(start[0]), init_world_z=float(start[1]),
                    init_direction=float(rng.uniform(0, 360)), target_x=int(target_pixel[0]),
                    target_y=int(target_pixel[1]), sampled_target_world_x=float(goal[0]),
                    sampled_target_world_z=float(goal[1]), euclidean_distance_m=distance,
                    map_route_distance_m=route_distance, detour_ratio=detour, difficulty=kind,
                    estimated_min_translation_steps=min_translation_steps, benchmark_step_budget=budget,
                    budget_probe_forward_step_m=step_m,
                    seed=int(seed + index + (100000 if split == "val" else 0)),
                    motion_random_seed=int(seed + index + (100000 if split == "val" else 0)),
                    dynamic_objects="moving", light_random_seed=0,
                    map_source="static_render_occupancy_not_physics_navmesh")
                if distance_quota:
                    record["sampling_distance_stratum"] = stratum
                if split == "train" and start_clearance_range is not None:
                    record["sampling_start_grid_clearance_m"] = float(clearance_m[si])
                    record["sampling_start_clearance_range_m"] = list(start_clearance_range)
                    record["sampling_clearance_source"] = "inflated_static_grid_not_physical_body_clearance"
                if split == "train" and goal_clearance_range is not None:
                    record["sampling_goal_grid_clearance_m"] = float(clearance_m[gi])
                    record["sampling_goal_clearance_range_m"] = list(goal_clearance_range)
                    record["sampling_clearance_source"] = "inflated_static_grid_not_physical_body_clearance"
                selected_records.append(record)
                counts[stratum] += 1
                # One pair per goal sample avoids a few goals dominating.
                break
        if sum(counts.values()) != count:
            raise ValueError(f"Could not meet {scene}/{split} quotas: {dict(counts)} / {quotas}; do not silently relax")
        if split == "val":
            for record in selected_records:
                reserved_endpoints.extend(((record["init_world_x"],record["init_world_z"]),
                    (record["sampled_target_world_x"],record["sampled_target_world_z"])))
        counts_report[split] = dict(counts)
    train_endpoints, val_endpoints = [], []
    for record in selected_records:
        bucket = train_endpoints if record["split"] == "train" else val_endpoints
        bucket.extend(((record["init_world_x"],record["init_world_z"]),
            (record["sampled_target_world_x"],record["sampled_target_world_z"])))
    all_endpoints = np.asarray(train_endpoints + val_endpoints)
    audit = dict(scene_name=scene, counts=counts_report,
        min_benchmark_endpoint_clearance_m=float(minimum_clearance(all_endpoints, canonical).min()),
        min_train_val_endpoint_clearance_m=float(minimum_clearance(np.asarray(train_endpoints),np.asarray(val_endpoints)).min()),
        unique_unordered_pairs=len({record_pair_key(r) for r in selected_records}),
        excluded_prior_pairs=len(excluded_pair_records or []),
        reserved_validation_reused=reserved_validation_records is not None,
        training_min_distance_m=training_min_distance_m,
        training_long_fraction=training_long_fraction,
        eligible_endpoint_cells=int(eligible.sum()),
        distances={split: dict(mean=float(np.mean([r["euclidean_distance_m"] for r in selected_records if r["split"]==split])),
                              max=float(max(r["euclidean_distance_m"] for r in selected_records if r["split"]==split)))
                   for split in ("train", "val")})
    if start_clearance_range is not None:
        audit["training_start_clearance_range_m"] = list(start_clearance_range)
        audit["training_start_grid_clearance_m"] = {
            "min": min(r["sampling_start_grid_clearance_m"] for r in selected_records if r["split"] == "train"),
            "max": max(r["sampling_start_grid_clearance_m"] for r in selected_records if r["split"] == "train"),
        }
        audit["training_start_eligible_cells_before_validation_exclusion"] = int(training_starts.sum())
        audit["clearance_source"] = "inflated_static_grid_not_physical_body_clearance"
        audit["sampling_map_obstacle_inflation_m"] = geometry.metadata.get("obstacle_clearance_m")
    if goal_clearance_range is not None:
        audit["training_goal_clearance_range_m"] = list(goal_clearance_range)
        audit["training_goal_grid_clearance_m"] = {
            "min": min(r["sampling_goal_grid_clearance_m"] for r in selected_records if r["split"] == "train"),
            "max": max(r["sampling_goal_grid_clearance_m"] for r in selected_records if r["split"] == "train"),
        }
        audit["training_goal_eligible_cells_before_validation_exclusion"] = int(training_goals.sum())
        audit["clearance_source"] = "inflated_static_grid_not_physical_body_clearance"
        audit["sampling_map_obstacle_inflation_m"] = geometry.metadata.get("obstacle_clearance_m")
    return selected_records, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--input-points", type=Path, default=Path("input_points.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", type=int, nargs="+", default=list(range(1,25)))
    parser.add_argument("--train-per-scene", type=int, default=300)
    parser.add_argument("--val-per-scene", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--reserved-validation-manifest", type=Path,
                        help="Reuse this entire held-out cohort, reserving both endpoint roles; do not resample validation.")
    parser.add_argument("--excluded-training-manifest", type=Path, action="append", default=[],
                        help="Avoid duplicating prior unordered training pairs; repeat for multiple manifests. Prior paths are never read.")
    parser.add_argument("--training-min-distance-m", type=float, default=8.0)
    parser.add_argument("--training-long-fraction", type=float, default=None,
                        help="Optional distance-based quotas (>=40m vs below40m), independent of detour labels.")
    parser.add_argument("--training-episode-prefix", default="independent_train")
    parser.add_argument("--training-start-clearance-m", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="Optional training-start distance range in meters on the already inflated approximate grid; validation and the default goal pool remain unchanged. Requires physical validation.")
    parser.add_argument("--training-goal-clearance-m", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="Optional training-goal distance range in meters on the already inflated approximate grid; validation and the default start pool remain unchanged. Requires physical validation.")
    parser.add_argument("--benchmark-clearance-m", type=float, default=4.0,
                        help="Includes 1m margin for bounded runtime calibration/pose differences")
    parser.add_argument("--split-clearance-m", type=float, default=3.0,
                        help="Includes 1m margin for the two runtime endpoint errors")
    args = parser.parse_args()
    try:
        validate_training_start_clearance(args.training_start_clearance_m)
        validate_training_clearance(args.training_goal_clearance_m, role="goal")
    except ValueError as error:
        parser.error(str(error))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("Refusing to overwrite an existing dataset; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    canonical = json.loads(args.input_points.read_text())
    reserved = ([json.loads(l) for l in args.reserved_validation_manifest.read_text().splitlines() if l.strip()]
                if args.reserved_validation_manifest else None)
    excluded = [json.loads(line) for path in args.excluded_training_manifest
                for line in path.read_text().splitlines() if line.strip()]
    all_records, audits = [], []
    for number in args.scenes:
        scene = f"scene{number}"
        geometry = NavigationMap.load(args.map_dir / scene)
        records, audit = sample_scene(geometry, canonical[scene], seed=args.seed+number*1009,
            train_count=args.train_per_scene, validation_count=args.val_per_scene,
            benchmark_clearance_m=args.benchmark_clearance_m, split_clearance_m=args.split_clearance_m,
            reserved_validation_records=([r for r in reserved if r["scene_name"] == scene] if reserved is not None else None),
            excluded_pair_records=[r for r in excluded if r["scene_name"] == scene],
            training_min_distance_m=args.training_min_distance_m,
            training_long_fraction=args.training_long_fraction,
            training_episode_prefix=args.training_episode_prefix,
            training_start_clearance_m=args.training_start_clearance_m,
            training_goal_clearance_m=args.training_goal_clearance_m)
        all_records.extend(records)
        audits.append(audit)
        print(json.dumps(audit), flush=True)
    for split in ("train", "val"):
        with (args.output_dir / f"{split}_manifest.jsonl").open("w") as stream:
            for record in all_records:
                if record["split"] == split:
                    stream.write(json.dumps(record,sort_keys=True)+"\n")
    # Fixed quick validation subset: one task from each difficulty per scene.
    with (args.output_dir / "val_quick_manifest.jsonl").open("w") as stream:
        for number in args.scenes:
            for kind in DIFFICULTIES:
                record = next(r for r in all_records if r["scene_name"]==f"scene{number}" and r["split"]=="val" and r["difficulty"]==kind)
                stream.write(json.dumps(record,sort_keys=True)+"\n")
    report = dict(format_version=1, scenes=audits,
        configured_benchmark_clearance_m=args.benchmark_clearance_m,
        configured_split_clearance_m=args.split_clearance_m,
        input_points_sha256=hashlib.sha256(args.input_points.read_bytes()).hexdigest(),
        reserved_validation_sha256=(hashlib.sha256(args.reserved_validation_manifest.read_bytes()).hexdigest()
                                    if args.reserved_validation_manifest else None),
        excluded_training_sha256=(hashlib.sha256(args.excluded_training_manifest[0].read_bytes()).hexdigest()
                                  if len(args.excluded_training_manifest) == 1 else None),
        provenance="independent free-space endpoints; source trajectories not read",
        limitation="occupancy connectivity and translation budget only; physical validation still required",
        train_pairs=sum(r["split"]=="train" for r in all_records), validation_pairs=sum(r["split"]=="val" for r in all_records))
    if args.training_start_clearance_m is not None:
        report["training_start_clearance_range_m"] = args.training_start_clearance_m
    if args.training_goal_clearance_m is not None:
        report["training_goal_clearance_range_m"] = args.training_goal_clearance_m
    if len(args.excluded_training_manifest) > 1:
        report["excluded_training_manifests_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.excluded_training_manifest}
    (args.output_dir / "sampling_audit.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
