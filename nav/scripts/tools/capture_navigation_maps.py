"""Capture calibrated scene geometry without collecting benchmark trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from nav.baselines.astar import AStarBaseline
from nav.baselines.rl.reward import compute_pointgoal_reward
from nav.data.navigation_map import NavigationMap
from nav.envs.unity_pointgoal import UnityPointGoalEnv
from nav.harness.coordinates import visual_to_world_coords, world_to_visual_coords


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Bootstrap poses for map calibration only")
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", type=int, nargs="+", default=list(range(1, 25)))
    parser.add_argument("--base-port", type=int, default=45000)
    parser.add_argument("--cell-m", type=float, default=0.5)
    parser.add_argument("--clearance-m", type=float, default=0.45)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    for scene_number in args.scenes:
        scene = f"scene{scene_number}"
        directory = args.output_dir / scene
        if (directory / "capture_audit.json").exists():
            print(f"Already captured {scene}", flush=True)
            continue
        # A sizable displacement on both axes gives a well-conditioned
        # two-point calibration. No source trajectory is read or followed.
        candidates = [t for t in tasks if t["scene_name"] == scene]
        task = max(candidates, key=lambda t: min(
            abs(t["init_world_x"] - t["sampled_target_world_x"]),
            abs(t["init_world_z"] - t["sampled_target_world_z"]),
        ))
        env = UnityPointGoalEnv(
            [task], unity_path=str(args.unity), output_dir=args.output_dir / "capture_unity",
            worker_id=scene_number, base_port=args.base_port, record_visuals=True,
            dynamic_objects="static", auto_reset=False, reward_fn=compute_pointgoal_reward,
        )
        try:
            env.reset()
            primed, rgb, projector = env.primed, env.minimap_rgb, env.minimap_projector
            assert primed is not None and rgb is not None and projector is not None
            def to_world(point):
                return visual_to_world_coords(primed.margin, point, projector, map_size=primed.minimap_size)
            def to_pixel(world):
                return world_to_visual_coords(primed.margin, *world, projector=projector,
                                              map_size=primed.minimap_size)
            extractor = AStarBaseline(grid_cell_m=args.cell_m,
                obstacle_clearance_m=args.clearance_m, minimum_obstacle_clearance_m=args.clearance_m,
                min_free_ratio=0.85, minimap_has_baked_markers=True, static_scene=True)
            walkable = extractor.build_navigation_grid(rgb, to_world,
                marker_points=(to_pixel(env.pose[:2]), to_pixel(env.target_world)))
            sx, sy = extractor.grid_step_x_px, extractor.grid_step_y_px
            origin = np.asarray(to_world((sx / 2, sy / 2)))
            basis = np.column_stack((np.asarray(to_world((sx * 1.5, sy / 2))) - origin,
                                     np.asarray(to_world((sx / 2, sy * 1.5))) - origin))
            metadata = dict(format_version=1, scene_name=scene, scene_id=scene_number-1,
                source="static_render_occupancy_not_physics_navmesh", world_origin=origin.tolist(),
                world_per_cell=basis.tolist(), grid_step_px=[sx, sy], margin=list(primed.margin),
                minimap_size=list(primed.minimap_size), projector=projector,
                obstacle_clearance_m=args.clearance_m, bootstrap_task=task,
                bootstrap_role="calibration_only_not_training_pair", unity=str(args.unity))
            geometry = NavigationMap(walkable, metadata)
            geometry.save(directory)
            cv2.imwrite(str(directory / "minimap.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            overlay = rgb.copy()
            blocked = cv2.resize((~walkable).astype(np.uint8), (walkable.shape[1]*sx, walkable.shape[0]*sy), interpolation=cv2.INTER_NEAREST)[:rgb.shape[0], :rgb.shape[1]].astype(bool)
            overlay[blocked] = (0.45 * overlay[blocked] + 0.55 * np.array([240, 50, 50])).astype(np.uint8)
            cv2.imwrite(str(directory / "occupancy_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            audit = dict(scene_name=scene, grid_shape=list(walkable.shape), free_cells=int(walkable.sum()),
                components=geometry.component_count-1, world_per_cell=basis.tolist(),
                actual_spawn=list(env.pose), actual_target=list(env.target_world),
                spawn_error_m=float(np.linalg.norm(np.asarray(env.pose[:2])-[task["init_world_x"],task["init_world_z"]])),
                minimap_sha256=hashlib.sha256((directory / "minimap.png").read_bytes()).hexdigest())
            (directory / "capture_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
            print(json.dumps(audit), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
