"""Distance, boundary, and no-expert regression checks for map-reward PPO."""

import tempfile
import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from nav.baselines.rl.regularization import reference_kl, reference_coefficient
from nav.data.navigation_map import NavigationMap
from nav.envs.unity_pointgoal import UnityPointGoalEnv, InvalidPointGoalSpawn
from nav.scripts.tools.sample_independent_pointgoal_pairs import minimum_clearance


def geometry(grid):
    return NavigationMap(grid, {"scene_name": "scene1", "world_origin": [0, 0],
                                "world_per_cell": [[1, 0], [0, 1]]})


class NavigationMapTest(unittest.TestCase):
    def test_missing_reset_coverage_retains_task_and_recovers_without_reward_jump(self):
        grid = np.ones((15, 15), bool)
        grid[:, 7] = False
        with tempfile.TemporaryDirectory() as directory:
            geometry(grid).save(Path(directory) / "scene1")
            task = dict(scene_name="scene1", episode_id="independent_train_0000",
                        init_world_x=4., init_world_z=3.,
                        sampled_target_world_x=12., sampled_target_world_z=3.)
            env = UnityPointGoalEnv([task], unity_path="unused", output_dir=directory,
                worker_id=0, base_port=48000, navigation_map_dir=directory,
                auto_reset=False, reward_fn=lambda *_a, **kw: kw["path_progress_m"], logger=Mock())
            env.task, env.pose, env.target_world = task, (4., 3., 0.), (12., 3.)
            env.primed = SimpleNamespace(env=Mock())
            env.depth_obs = np.ones((1, 24, 32), np.float32)
            env.distance_m = env.best_distance_m = 8.
            env._reset_map_reward()
            self.assertIsNone(env.map_distance_m)
            self.assertIs(env.task, task)
            env.logger.warning.assert_called_once()
            def observe(x):
                env.pose = (x, 3., 0.)
                env.distance_m = 12. - x
                return {"goal": np.array([.12, 0., 1.], dtype=np.float32)}
            # Simulated entry into mapped space establishes a baseline, not a
            # reward jump. Only a subsequent known-to-known step earns progress.
            with patch.object(env, "_read_observation", side_effect=lambda: observe(9.)):
                _, reward, _, info = env.step(0)
            self.assertEqual(reward, 0.)
            self.assertFalse(info["map_progress_valid"])
            self.assertIsNotNone(env.best_map_distance_m)
            with patch.object(env, "_read_observation", side_effect=lambda: observe(10.)):
                _, reward, _, info = env.step(0)
            self.assertGreater(reward, 0.)
            self.assertTrue(info["map_progress_valid"])
            self.assertEqual(env.steps_without_progress, 0)
            env.target_world = (100., 100.)
            env.task = dict(task, sampled_target_world_x=100., sampled_target_world_z=100.)
            env.pose = (4., 3., 0.)
            env._reset_map_reward()
            self.assertTrue(np.isinf(env.navigation_distance_field).all())
            self.assertIsNone(env.map_distance_m)

    def test_invalid_spawn_retries_are_training_only_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            env=UnityPointGoalEnv([{"scene_name":"scene1"}],unity_path="unused",
                output_dir=directory,worker_id=0,base_port=48000,reward_fn=Mock(),
                resample_invalid_spawns=True,logger=Mock())
            with patch.object(env,"_reset_once",side_effect=[InvalidPointGoalSpawn("displaced"),{"depth":"ok"}]) as reset:
                self.assertEqual(env.reset(),{"depth":"ok"})
                self.assertEqual(reset.call_count,2)
            self.assertEqual(env.invalid_spawn_resets,1)
            self.assertTrue((Path(directory)/"invalid_spawns_env0.jsonl").is_file())
            env.resample_invalid_spawns=False
            with patch.object(env,"_reset_once",side_effect=InvalidPointGoalSpawn("displaced")) as reset:
                with self.assertRaises(InvalidPointGoalSpawn):
                    env.reset()
                self.assertEqual(reset.call_count,1)

    def test_diagonal_cannot_cross_a_blocked_corner(self):
        navmap = geometry(np.eye(3, dtype=bool))
        field = navmap.distance_field((0, 0))
        self.assertTrue(np.isinf(field[1, 1]))
        self.assertTrue(np.isinf(field[2, 2]))
        self.assertIsNone(navmap.distance_at((1,1),field,max_snap_m=3))

    def test_meter_based_fringe_does_not_depend_on_one_grid_cell(self):
        grid=np.ones((12,12),bool)
        grid[:,5:8]=False
        navmap=NavigationMap(grid,{"world_origin":[0,0],"world_per_cell":[[.2,0],[0,.2]]})
        field=navmap.distance_field((2.0,1.0))
        self.assertIsNotNone(navmap.distance_at((1.35,1.0),field,max_snap_m=.6))
        # The nearer disconnected side must not be skipped for the goal side.
        self.assertIsNone(navmap.distance_at((1.05,1.0),field,max_snap_m=.8))

    def test_correct_detour_progresses_while_euclidean_distance_increases(self):
        grid = np.ones((10, 10), dtype=bool)
        grid[0:8, 5] = False
        navmap = geometry(grid)
        goal = (7, 2)
        field = navmap.distance_field(goal)
        first, second = (3, 3), (3, 4)
        self.assertGreater(np.linalg.norm(np.asarray(second)-goal), np.linalg.norm(np.asarray(first)-goal))
        self.assertLess(navmap.distance_at(second, field), navmap.distance_at(first, field))
        self.assertEqual(navmap.distance_at(first, field), navmap.distance_at(first, field))

    def test_distance_is_in_meters_and_serialization_is_exact(self):
        navmap = NavigationMap(np.ones((5,5),bool), {"world_origin":[4,8],"world_per_cell":[[0,-2],[0.5,0]]})
        goal = navmap.cell_to_world((4,4))
        field = navmap.distance_field(goal)
        self.assertAlmostEqual(field[4,0], 2.0)
        self.assertAlmostEqual(field[0,4], 8.0)
        self.assertIsNone(navmap.distance_at((100,100), field))
        with tempfile.TemporaryDirectory() as directory:
            navmap.save(Path(directory))
            loaded = NavigationMap.load(directory)
            np.testing.assert_array_equal(field, loaded.distance_field(goal))

    def test_reference_kl_is_exact_finite_and_reference_has_no_gradient(self):
        logits = torch.tensor([[0.4, -0.2, 0.8]], requires_grad=True)
        reference = torch.tensor([[0.1, 0.5, -0.7]], requires_grad=True)
        loss = reference_kl(logits, reference)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=logits),
            torch.distributions.Categorical(logits=reference))
        torch.testing.assert_close(loss, expected)
        loss.sum().backward()
        self.assertIsNone(reference.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        torch.testing.assert_close(reference_kl(logits, logits), torch.zeros(1))
        self.assertEqual(reference_coefficient(1, 0.1, 0.02, 100), 0.1)
        self.assertAlmostEqual(reference_coefficient(101, 0.1, 0.02, 100), 0.02)
        self.assertAlmostEqual(reference_coefficient(201, 0.1, 0.02, 100), 0.02)

    def test_cross_role_endpoint_clearance(self):
        points = np.array([[0.,0.], [1.,0.], [6.,0.]])
        excluded = np.array([[0.,0.], [8.,0.]])
        np.testing.assert_array_equal(minimum_clearance(points, excluded), [0,1,2])

    def test_map_reward_does_not_construct_or_query_a_teacher(self):
        grid = np.ones((15,15),bool)
        grid[:11,7] = False
        navmap = geometry(grid)
        task = dict(scene_id=0,scene_name="scene1",episode_id="independent_train_0000",
                    init_world_x=4.,init_world_z=3.,sampled_target_world_x=10.,sampled_target_world_z=3.)
        with tempfile.TemporaryDirectory() as directory:
            navmap.save(Path(directory)/"scene1")
            env = UnityPointGoalEnv([task], unity_path="unused", output_dir=directory,
                worker_id=0,base_port=48000,navigation_map_dir=directory,auto_reset=False,
                reward_fn=lambda *_a, **kw: kw["path_progress_m"],logger=Mock())
            env.task=task
            env.pose=(4.,3.,0.)
            env.target_world=(10.,3.)
            env.primed=SimpleNamespace(env=Mock())
            env.depth_obs=np.ones((1,24,32),np.float32)
            env.distance_m=env.best_distance_m=6.
            env._reset_map_reward()
            old=env.map_distance_m
            env.steps_without_progress=20
            def observe():
                env.pose=(4.,4.,0.)
                env.distance_m=float(np.hypot(6,1))
                return {"goal":np.array([.12,0.,1.],dtype=np.float32)}
            with patch.object(env,"_read_observation",side_effect=observe), patch.object(
                env,"predict_astar_teacher_action",side_effect=AssertionError("No teacher allowed")):
                _,reward,_,info=env.step(0)
            self.assertGreater(reward,0)
            self.assertLess(env.map_distance_m,old)
            self.assertEqual(env.steps_without_progress,0)
            self.assertTrue(info["map_progress_valid"])
            self.assertIsNone(env.astar_teacher)
            self.assertFalse(env.online_astar_teacher)

    def test_full_reference_ppo_matches_serial_and_parallel_steps(self):
        from nav.baselines.rl import trainer
        from nav.models.policies import DaggerTransformerActorCritic, dagger_ppo_config
        from nav.scripts.rl.train_pointgoal_ppo import parse_args

        torch.set_num_threads(1)
        config=dagger_ppo_config(dict(policy_type="transformer",use_depth=True,use_rgb=False,
            navigation_only_actions=True,chunk_size=1,goal_encoding="unity_egocentric_v2",
            goal_rep="polar",goal_distance_scale_m=50.,seq_len=6,num_layers=1,
            depth_backbone="resnet18",img_size=32,half_width=False,goal_action_residual=False))

        class Env:
            def __init__(self,tasks,**kwargs):
                self.index=0
            def reset(self):
                return {"depth":np.full((1,32,32),.1,np.float32),
                        "goal":np.array([.2,0.,1.],np.float32)}
            def step(self,action):
                self.index+=1
                return self.reset(), .1+action*.05, False, {
                    "executed_action_index":action,"shield_intervened":False,
                    "map_progress_valid":True}
            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            initial=root/"initial.pt"
            reference=root/"reference.pt"
            actor=DaggerTransformerActorCritic(config)
            torch.save({"model":actor.state_dict(),"model_config":asdict(config)},initial)
            with torch.no_grad():
                next(actor.head.parameters())[0, 0].add_(.8)
            torch.save({"model":actor.state_dict(),"model_config":asdict(config)},reference)
            manifest=root/"tasks.jsonl"
            manifest.write_text("".join(json.dumps({"scene_name":f"scene{i}","split":"train"})+"\n" for i in (1,2)))
            unity=root/"unity"
            unity.touch()
            results=[]
            for temperature, parallel in ((1.,False),(1.,True),(.5,False),(.5,True)):
                output=root/f"{temperature}_{parallel}"
                argv=["train","--manifest",str(manifest),"--unity",str(unity),
                    "--output-dir",str(output),"--init-ppo-checkpoint",str(initial),
                    "--reference-policy-checkpoint",str(reference),"--reference-kl-coef","0.1",
                    "--navigation-map-dir",str(root),"--device","cpu","--num-envs","2",
                    "--total-updates","2","--rollout-steps","2","--bptt-len","1",
                    "--chunks-per-minibatch","4","--ppo-epochs","1","--critic-warmup-updates","1",
                    "--policy-temperature",str(temperature)]
                if parallel:
                    argv.append("--parallel-env-steps")
                with patch.object(sys,"argv",argv):
                    args=parse_args()
                with patch.object(trainer,"UnityPointGoalEnv",Env),patch.object(trainer.logging,"basicConfig"),patch.object(
                    trainer.logging,"FileHandler",return_value=trainer.logging.NullHandler()):
                    trainer.run_training(args)
                rows=[json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines()]
                self.assertTrue(all(r["reference_kl"]>0 for r in rows))
                self.assertTrue(all(r["teacher_action_fraction"]==0 and r["teacher_coefficient"]==0 for r in rows))
                self.assertTrue(all(r["map_progress_coverage"]==1 for r in rows))
                self.assertTrue(all(r["policy_temperature"]==temperature for r in rows))
                # Each update uses one minibatch/epoch. Before that sole
                # optimizer step the behavior and replay likelihoods match,
                # including when T != 1; inconsistent scaling breaks this.
                self.assertTrue(all(abs(r["approx_kl"])<1e-6 for r in rows))
                self.assertTrue(all(abs(r["sampled_kl_k3"])<1e-6 for r in rows))
                self.assertTrue(all(r["maximum_checked_kl"]<1e-6 and r["ppo_clip_fraction"]==0 for r in rows))
                self.assertEqual(rows[0]["gradient_norm_policy"], 0.)
                self.assertTrue(rows[1]["gradient_norm_policy"] > 0.)
                self.assertTrue(all(r["gradient_norm_total"] >= r["gradient_norm_value"] > 0 for r in rows))
                results.append(torch.load(output/"latest.pt",map_location="cpu",weights_only=False)["model"])
            for first, second in ((0,1),(2,3)):
                for name in results[first]:
                    torch.testing.assert_close(results[first][name],results[second][name],atol=0,rtol=0)
            # Reuse the final run's optimizer moments while changing only its
            # configured actor learning rate; the third update must not reset.
            previous=torch.load(output/"latest.pt",map_location="cpu",weights_only=False)
            with patch.object(sys,"argv",argv+["--resume","--resume-optimizer-lr-mode","config",
                    "--total-updates","3","--policy-lr-scale","0.05"]):
                args=parse_args()
            with patch.object(trainer,"UnityPointGoalEnv",Env),patch.object(trainer.logging,"basicConfig"),patch.object(
                trainer.logging,"FileHandler",return_value=trainer.logging.NullHandler()):
                trainer.run_training(args)
            resumed=torch.load(output/"latest.pt",map_location="cpu",weights_only=False)
            self.assertEqual(resumed["update"],3)
            rates={g["group_name"]:g["lr"] for g in resumed["optimizer"]["param_groups"]}
            self.assertAlmostEqual(rates["policy_temporal"],args.lr*.05)
            self.assertAlmostEqual(rates["value"],args.lr)
            for key, state in previous["optimizer"]["state"].items():
                self.assertEqual(resumed["optimizer"]["state"][key]["step"].item(),state["step"].item()+1)


if __name__ == "__main__":
    unittest.main()
