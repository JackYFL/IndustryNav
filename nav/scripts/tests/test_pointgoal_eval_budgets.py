"""Pinned original horizons must survive reset jitter without changing actions."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from nav.eval.budgets import (
    apply_reference_budget, load_reference_budgets, validate_completed_budgets,
)
from nav.scripts.evaluation import evaluate_pointgoal_policy as evaluation


class PointGoalEvalBudgetsTest(unittest.TestCase):
    def test_hash_bound_budget_selection_ignores_success_labels_and_supports_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"reference.json"
            rows = [dict(scene_name="scene1", episode_id=f"point{i}", step_budget=120+i,
                         success=bool(i%2), distance_world=-999) for i in (1,2)]
            path.write_text(json.dumps({"episodes": rows}))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            budgets, found = load_reference_budgets(path, digest, [("scene1", "point2")])
            self.assertEqual(budgets, {("scene1", "point2"):122})
            self.assertEqual(found, digest)
            with self.assertRaisesRegex(ValueError, "hash"):
                load_reference_budgets(path, "0"*64, [("scene1", "point2")])
            for keys in ([], [("scene9", "point1")], [("scene1", "point2")]*2):
                with self.assertRaises(ValueError): load_reference_budgets(path, digest, keys)

    def test_reject_duplicate_or_invalid_reference_budgets(self):
        row = dict(scene_name="scene1", episode_id="point1", step_budget=120)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"reference.json"
            for rows in ([row,row],*[ [dict(row,step_budget=b)] for b in (True,79,321,120.0)]):
                p.write_text(json.dumps({"episodes":rows}));digest=hashlib.sha256(p.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):load_reference_budgets(p,digest,[("scene1","point1")])

    def test_apply_before_actions_only_and_default_noop(self):
        task=dict(scene_name="scene22",episode_id="point4")
        env=SimpleNamespace(episode_steps=0,episode_max_steps=126)
        self.assertIsNone(apply_reference_budget(env,task,None))
        self.assertEqual(env.episode_max_steps,126)
        self.assertEqual(apply_reference_budget(env,task,{("scene22","point4"):127}),126)
        self.assertEqual(env.episode_max_steps,127)
        env.episode_steps=1
        with self.assertRaises(ValueError):apply_reference_budget(env,task,{("scene22","point4"):128})
        self.assertEqual(env.episode_max_steps,127)

    def test_resume_cannot_silently_keep_wrong_budget_duplicate_or_unknown_task(self):
        rows=[dict(scene_name="scene1",episode_id="point1",step_budget=120)]
        budgets={("scene1","point1"):120}
        validate_completed_budgets(rows,budgets)
        validate_completed_budgets(rows,None)
        for old in (rows*2,[dict(rows[0],step_budget=121)],[dict(rows[0],step_budget=120.0)],
                    [dict(rows[0],episode_id="missing")]):
            with self.assertRaises(ValueError):validate_completed_budgets(old,budgets)

    def args(self, root, extra=()):
        with patch.object(sys,"argv",["eval","--input-points","input_points.json",
            "--checkpoint","missing.pt","--unity","missing_unity","--baseline","ppo",
            "--persistent-policy","--output-root",str(root),*extra]):
            return evaluation.parse_args()

    def test_persistent_evaluator_uses_fixed_horizon_at_every_reset_without_policy_input_change(self):
        class Controller:
            config=SimpleNamespace(absolute_scene_state=False,num_scenes=24,world_coordinate_scale_m=50.)
            def __init__(self,*a,**k):pass
            def reset(self):pass
            def predict_action(self,**state):
                assert set(state)=={"depth_obs","curr_world_x","curr_world_z","curr_yaw_deg","target_world_x","target_world_z","scene_id"}
                return "forward"
        class Env:
            def __init__(self,tasks,**kwargs):
                assert kwargs["dynamic_objects"] == (mode or "moving")
                self.tasks=iter(tasks);self.index=-1;self.closed=False
            def reset(self):
                self.task=next(self.tasks);self.index+=1;self.episode_steps=0
                self.episode_max_steps=(126,130)[self.index];self.pose=(0.,0.,0.)
                self.target_world=(10.,0.);self.depth_obs=np.zeros((4,4,1),np.float32)
                self.episode_collisions=0;self.steps_without_progress=0;self.distance_m=10.
                return {"goal":np.array([.2,0.,1.])}
            def step(self,action):
                assert action==0;self.episode_steps+=1
                return {"goal":np.array([.2,0.,1.])},0.,self.episode_steps>=self.episode_max_steps,dict(
                    success=False,collision=False,distance_m=10.,episode_steps=self.episode_steps,
                    episode_max_steps=self.episode_max_steps,episode_collisions=0,
                    episode_collision_events=0,episode_warnings=0)
            def close(self):self.closed=True
        tasks=[dict(scene_name="scene1",scene_id=1,episode_id=f"point{i}") for i in (1,2)]
        for pinned, mode in ((False, None), (True, None), (False, "static"), (True, "static")):
            with self.subTest(pinned=pinned,mode=mode),tempfile.TemporaryDirectory() as directory:
                args=self.args(Path(directory), ["--dynamic-objects", mode] if mode else [])
                if pinned:
                    args._reference_budgets={("scene1","point1"):127,("scene1","point2"):129}
                    args._budget_reference_sha256="pinned-test-reference"
                with patch("nav.baselines.rl.agent.PPOPointGoalController",Controller),patch(
                    "nav.envs.unity_pointgoal.UnityPointGoalEnv",Env),redirect_stdout(io.StringIO()):
                    rows=evaluation.evaluate_persistent_worker(args,tasks,0)
                expected=[127,129] if pinned else [126,130]
                self.assertEqual([r["step_budget"] for r in rows],expected)
                self.assertEqual([r["steps_taken"] for r in rows],expected)
                self.assertTrue(all(r["dynamic_objects"] == (mode or "moving") for r in rows))
                self.assertTrue(all(r["executed_action_counts"]=={"forward":n} for r,n in zip(rows,expected)))
                if pinned:self.assertEqual([r["reset_dynamic_step_budget"] for r in rows],[126,130])
                else:self.assertTrue(all("reset_dynamic_step_budget" not in r for r in rows))

    def test_static_cli_reaches_task_commands_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, ["--dynamic-objects", "static", "--scenes", "1"])
            args.checkpoint = root / "checkpoint.pt"
            args.checkpoint.touch()
            args.input_points = root / "input_points.json"
            args.input_points.write_text(json.dumps({"scene1": [{
                "point_id": "point1", "start": {"x": 1, "z": 2, "direction": 0},
                "target": {"x": 100, "y": 200},
            }]}))
            def run(actual_args, tasks):
                self.assertEqual(tasks[0]["dynamic_objects"], "static")
                command = evaluation.command_for(actual_args, tasks[0], 0, root / "episode")
                self.assertEqual(command[command.index("--dynamic_objects") + 1], "static")
                return [dict(scene_name="scene1", episode_id="point1", success=True)]
            with patch.object(evaluation, "parse_args", return_value=args), patch.object(
                evaluation, "evaluate_persistent", side_effect=run,
            ), redirect_stdout(io.StringIO()):
                evaluation.main()
            self.assertEqual(json.loads((root / "summary.json").read_text())["dynamic_objects"], "static")

    def test_resume_rejects_results_from_different_motion_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, ["--dynamic-objects", "static", "--resume"])
            (root / "worker_0.jsonl").write_text(json.dumps(dict(
                scene_name="scene1", episode_id="point1", dynamic_objects="moving",
            )) + "\n")
            with self.assertRaisesRegex(ValueError, "different dynamic-objects mode"):
                evaluation.evaluate_persistent_worker(args, [], 0)

    def test_invalid_reference_cli_combination_fails_before_unity(self):
        with tempfile.TemporaryDirectory() as directory:
            for extra,remove_persistent in ((["--step-budget-reference","reference.json"],False),
                    (["--step-budget-reference-sha256","a"*64],False),
                    (["--step-budget-reference","reference.json","--step-budget-reference-sha256","a"*64],True)):
                args=self.args(Path(directory),extra)
                if remove_persistent:args.persistent_policy=False
                with patch.object(evaluation,"parse_args",return_value=args),patch.object(evaluation,"evaluate_persistent") as run:
                    with self.assertRaises(SystemExit):evaluation.main()
                    run.assert_not_called()


if __name__=="__main__":unittest.main()
