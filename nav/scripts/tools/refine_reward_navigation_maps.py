"""Derive finer reward-only maps from the immutable calibrated scene renders.

Sampling keeps its conservative clearance. Reward coverage needs to include
physical states in that clearance fringe; the online collision cost still
penalizes contact. This does not modify scene images or sampling endpoints.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from nav.baselines.astar import AStarBaseline
from nav.data.navigation_map import NavigationMap
from nav.harness.coordinates import visual_to_world_coords, world_to_visual_coords


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--cell-m",type=float,default=.3)
    parser.add_argument("--clearance-m",type=float,default=.1)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("Refusing to overwrite existing reward maps")
    for number in range(1,25):
        scene=f"scene{number}"
        directory=args.source_dir/scene
        meta=json.loads((directory/"metadata.json").read_text())
        audit=json.loads((directory/"capture_audit.json").read_text())
        rgb=cv2.cvtColor(cv2.imread(str(directory/"minimap.png")),cv2.COLOR_BGR2RGB)
        def to_world(point):
            return visual_to_world_coords(meta["margin"],point,meta["projector"],map_size=meta["minimap_size"])
        def to_pixel(world):
            return world_to_visual_coords(meta["margin"],*world,projector=meta["projector"],map_size=meta["minimap_size"])
        extractor=AStarBaseline(grid_cell_m=args.cell_m,obstacle_clearance_m=args.clearance_m,
            minimum_obstacle_clearance_m=args.clearance_m,min_free_ratio=.7,
            minimap_has_baked_markers=True,static_scene=True)
        grid=extractor.build_navigation_grid(rgb,to_world,marker_points=(
            to_pixel(audit["actual_spawn"][:2]),to_pixel(audit["actual_target"])))
        sx,sy=extractor.grid_step_x_px,extractor.grid_step_y_px
        origin=np.asarray(to_world((sx/2,sy/2)))
        basis=np.column_stack((np.asarray(to_world((1.5*sx,sy/2)))-origin,
                               np.asarray(to_world((sx/2,1.5*sy)))-origin))
        meta.update(world_origin=origin.tolist(),world_per_cell=basis.tolist(),grid_step_px=[sx,sy],
            obstacle_clearance_m=args.clearance_m,role="reward_only_not_sampling",
            source_minimap_sha256=hashlib.sha256((directory/"minimap.png").read_bytes()).hexdigest())
        geometry=NavigationMap(grid,meta)
        output=args.output_dir/scene
        geometry.save(output)
        cv2.imwrite(str(output/"walkable.png"),grid.astype(np.uint8)*255)
        print(json.dumps(dict(scene=scene,free_cells=int(grid.sum()),components=geometry.component_count-1)),flush=True)


if __name__=="__main__":
    main()
