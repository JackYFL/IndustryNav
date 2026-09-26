"""Physically check sampled poses/targets in Unity without collecting labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from nav.baselines.rl.reward import compute_pointgoal_reward
from nav.config import BEHAVIOR_NAME
from nav.envs.unity_pointgoal import UnityPointGoalEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", type=int, nargs="+", default=list(range(1,25)))
    parser.add_argument("--per-scene", type=int, default=0, help="0 checks every task")
    parser.add_argument("--base-port", type=int, default=44000)
    args=parser.parse_args()
    records=[json.loads(line) for path in args.manifest for line in path.read_text().splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for number in args.scenes:
        scene=f"scene{number}"
        tasks=[r for r in records if r["scene_name"]==scene]
        if args.per_scene:
            # Include every split/difficulty before taking additional records.
            groups={}
            for r in tasks:
                groups.setdefault((r["split"],r["difficulty"]),[]).append(r)
            tasks=[group[i] for i in range(max(map(len,groups.values()))) for group in groups.values() if i<len(group)][:args.per_scene]
        output=args.output_dir/f"{scene}.jsonl"
        done={}
        if output.exists():
            done={r["episode_id"]:r for r in map(json.loads,output.read_text().splitlines())}
        tasks=[t for t in tasks if t["episode_id"] not in done]
        if not tasks:
            continue
        env=UnityPointGoalEnv(tasks,unity_path=str(args.unity),output_dir=args.output_dir/"unity",
            worker_id=number,base_port=args.base_port,task_sampling="sequential",auto_reset=False,
            navigation_map_dir=args.map_dir,reward_fn=compute_pointgoal_reward,
            preserve_depth_precision=True,stuck_recovery_steps=0)
        try:
            with output.open("a") as stream:
                for task in tasks:
                    try:
                        env.reset()  # Check physical placement and calibration.
                    except RuntimeError as exc:
                        if not str(exc).startswith("Sampled spawn disagrees"):
                            raise
                        failure=dict(scene_name=scene,episode_id=task["episode_id"],
                            split=task["split"],difficulty=task["difficulty"],status="invalid_spawn",
                            error=str(exc),actual_pose=list(env.pose) if env.pose else None)
                        stream.write(json.dumps(failure)+"\n")
                        stream.flush()
                        print(json.dumps(failure),flush=True)
                        continue
                    if env.task["episode_id"]!=task["episode_id"]:
                        raise RuntimeError("Unexpected task skip during physical validation")
                    decision,_=env.primed.env.get_steps(BEHAVIOR_NAME)
                    height=float(decision.obs[3][0,1])
                    initial_pose=env.pose
                    target=env.target_world
                    initial_distance=env.distance_m
                    observation,reward,_,info=env.step(0)
                    result=dict(scene_name=scene,episode_id=task["episode_id"],split=task["split"],
                        status="checked",
                        difficulty=task["difficulty"],initial_pose=list(initial_pose),initial_height=height,
                        actual_target=list(target),initial_distance_m=initial_distance,
                        spawn_error_m=float(np.linalg.norm(np.asarray(initial_pose[:2])-[task["init_world_x"],task["init_world_z"]])),
                        target_error_m=float(np.linalg.norm(np.asarray(target)-[task["sampled_target_world_x"],task["sampled_target_world_z"]])),
                        forward_displacement_m=info["actual_displacement_m"],collision=info["collision"],
                        warning=info["warning"],map_progress_valid=info["map_progress_valid"],
                        finite_depth=bool(np.isfinite(observation["depth"]).all()))
                    if not result["finite_depth"] or not np.isfinite(height):
                        raise RuntimeError("Invalid physical observation")
                    stream.write(json.dumps(result)+"\n")
                    stream.flush()
            print(json.dumps({"scene":scene,"checked":len(tasks),"status":"completed"}),flush=True)
        finally:
            env.close()


if __name__=="__main__":
    main()
