"""Shielded PPO uses the unchanged benchmark action mapping, without an expert."""

import ast
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from nav.envs.unity_pointgoal import UnityPointGoalEnv
from nav.safety.pointgoal_protocol import PointGoalSafetyProtocol, PointGoalShieldConfig
from nav.safety.policy_shield import ReactiveSafetyShield


class TrainingShieldTest(unittest.TestCase):
    @staticmethod
    def _live_evaluator_blocks():
        # Exercise the *actual*, unchanged evaluation branches, rather than a
        # hand-copied reference that could drift alongside the implementation.
        path = Path(__file__).parents[1] / "evaluation/evaluate_pointgoal_policy.py"
        tree = ast.parse(path.read_text())
        before = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "shield is not None"
            and any(
                isinstance(child, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "forward_warning" for t in child.targets)
                for child in node.body
            )
        ]
        after = []
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            for index, child in enumerate(body[:-1]):
                if (
                    isinstance(child, ast.Assign)
                    and ast.unparse(child) == "collision_streak = collision_streak + 1 if info['collision'] else 0"
                ):
                    after.append([child, body[index + 1]])
        if len(before) != 1 or len(after) != 1:
            raise AssertionError("Canonical evaluation shield branches changed; review parity test")
        def code(nodes):
            return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec")
        return code(before), code(after[0])

    def test_training_protocol_matches_live_benchmark_state_machine(self):
        before, after = self._live_evaluator_blocks()
        config = PointGoalShieldConfig()
        args = SimpleNamespace(**{
            "shield_" + name: value
            for name, value in vars(config).items() if name != "move_command"
        }, shield_terminal_homing_distance_m=0.0, fallback_turn_final_stage_only=False,
            fallback_turn_streak=0, fallback_turn_escape_steps=4)
        clear = np.full((48, 64), 4.0, dtype=np.float32)
        blocked = np.full_like(clear, 0.2)
        half_blocked = clear.copy()
        half_blocked[:, :32] = 0.2
        rng = np.random.default_rng(831)
        for phase in range(3):
            with self.subTest(phase=phase):
                protocol = PointGoalSafetyProtocol(config)
                reference = dict(
                    args=args, shield=ReactiveSafetyShield(max_clearance_turns=8),
                    decode_depth_observation_meters=lambda depth: depth,
                    collision_streak=0, warning_streak=0, turn_streak=0,
                    turn_escape_remaining=0, turn_escape_interventions=0,
                    proactive_warning_recoveries=0, terminal_homing_interventions=0,
                    fallback_active=False, second_fallback_active=False,
                )
                for step in range(500):
                    # Dedicated prefixes hit physical contact, warning-driven
                    # recovery, clear-space turn escape, then random sequences.
                    if step < 8:
                        proposed, depth = "forward", clear
                    elif step < 40:
                        proposed, depth = "forward", blocked
                    elif step < 100:
                        proposed, depth = "turn left", clear
                    else:
                        proposed = ("forward", "turn right", "turn left")[rng.integers(3)]
                        depth = (clear, blocked, half_blocked)[rng.integers(3)]
                    bearing = float(rng.uniform(-1, 1))
                    reference.update(
                        action_name=proposed, intervened=False,
                        env=SimpleNamespace(depth_obs=depth, distance_m=12.0),
                        observation={"goal": [0.24, bearing, 0.5]},
                    )
                    exec(before, reference)
                    actual, intervened = protocol.filter_action(
                        proposed, depth_m=depth, goal_bearing_sin=bearing,
                    )
                    self.assertEqual((actual, intervened), (reference["action_name"], reference["intervened"]))
                    collision = actual == "forward" and (step < 8 or rng.random() < 0.4)
                    reference["info"] = {"collision": collision}
                    exec(after, reference)
                    protocol.observe_step(collision=collision, depth_m=depth, goal_bearing_sin=bearing)
                    for name in (
                        "collision_streak", "warning_streak", "turn_streak",
                        "turn_escape_remaining", "turn_escape_interventions",
                        "proactive_warning_recoveries",
                    ):
                        self.assertEqual(getattr(protocol, name), reference[name], (step, name))
                    for name in (
                        "last_turn", "interventions", "recovery_turns_remaining",
                        "recovery_forwards_remaining", "clearance_turns_remaining",
                    ):
                        self.assertEqual(getattr(protocol.reactive, name), getattr(reference["shield"], name), (step, name))
                self.assertGreater(protocol.interventions, 0)
                self.assertGreater(protocol.proactive_warning_recoveries, 0)
                self.assertGreater(protocol.turn_escape_interventions, 0)
                protocol.reset()
                self.assertEqual(protocol.interventions, 0)
                self.assertEqual(protocol.collision_streak, 0)

    def test_unity_environment_reports_proposed_and_executed_actions(self):
        task = dict(scene_id=0, scene_name="scene1", episode_id="new01")
        env = UnityPointGoalEnv(
            [task], unity_path="unity", output_dir="unused", worker_id=0,
            base_port=50000, training_safety_shield=True, auto_reset=False,
            reward_fn=lambda *_a, **_kw: 0.0, logger=Mock(),
        )
        env.task = task
        env.primed = SimpleNamespace(env=Mock())
        env.pose, env.target_world = (0.0, 0.0, 0.0), (0.0, 10.0)
        env.depth_obs = np.full((48, 64, 1), 0.1, dtype=np.float32)
        env.distance_m = env.best_distance_m = env.episode_initial_distance_m = 10.0
        env.training_shield.reactive.last_turn = "turn right"
        env.training_shield.reactive.recovery_turns_remaining = 1
        def observe():
            env.pose = (0.0, 0.0, 22.5)
            return {"goal": np.array([0.2, -0.3, 0.9], dtype=np.float32)}
        with patch.object(env, "_read_observation", side_effect=observe), patch(
            "nav.envs.unity_pointgoal.decode_depth_observation_meters",
            return_value=np.full((48, 64), 4.0, dtype=np.float32),
        ):
            _, _, done, info = env.step(0)  # Policy proposes forward.
        self.assertFalse(done)
        self.assertEqual(info["proposed_action_index"], 0)
        self.assertEqual(info["executed_action_index"], 1)
        self.assertTrue(info["shield_intervened"])
        self.assertEqual(env.previous_action_name, "turn right")
        self.assertEqual(env.episode_rotation_steps, 1)
        self.assertEqual(env.episode_forward_steps, 0)
        self.assertFalse(info["collision"])

    def test_ppo_trainer_scores_proposals_but_remembers_executed_actions(self):
        from nav.baselines.rl import trainer
        from nav.models.policies import DaggerTransformerActorCritic, dagger_ppo_config
        from nav.scripts.rl.train_pointgoal_ppo import parse_args

        torch.set_num_threads(1)
        histories, likelihood_actions, environments = [], [], []
        real_factory = trainer.build_pointgoal_actor_critic
        real_distribution = trainer.Categorical

        def model_factory(config):
            model = real_factory(config)
            model.register_forward_pre_hook(
                lambda _model, inputs: histories.append(inputs[2].detach().clone())
            )
            return model

        def distribution_factory(*args, **kwargs):
            distribution = real_distribution(*args, **kwargs)
            log_prob = distribution.log_prob
            def tracked(value):
                likelihood_actions.append(value.detach().clone())
                return log_prob(value)
            distribution.log_prob = tracked
            return distribution

        class ShieldedEnv:
            def __init__(self, tasks, **kwargs):
                assert kwargs["training_safety_shield"]
                self.proposals = []
                self.closed = False
                environments.append(self)

            def reset(self):
                return {"depth": np.full((1, 32, 32), 0.1, dtype=np.float32),
                        "goal": np.array([0.2, 0.0, 1.0], dtype=np.float32)}

            def step(self, action):
                self.proposals.append(action)
                return self.reset(), 0.1, False, {
                    "executed_action_index": 1, "shield_intervened": True,
                }

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "tasks.jsonl"
            manifest.write_text(json.dumps({"scene_name": "scene1", "split": "train"}) + "\n")
            unity = root / "unity"
            unity.touch()
            config = dagger_ppo_config(dict(
                policy_type="transformer", use_depth=True, use_rgb=False,
                navigation_only_actions=True, chunk_size=1,
                goal_encoding="unity_egocentric_v2", goal_rep="polar",
                goal_distance_scale_m=50.0, seq_len=6, num_layers=1,
                depth_backbone="resnet18", img_size=32, half_width=False,
                goal_action_residual=False,
            ))
            checkpoint = root / "initial.pt"
            torch.save({"model": DaggerTransformerActorCritic(config).state_dict(),
                        "model_config": asdict(config)}, checkpoint)
            output = root / "run"
            argv = [
                "train", "--manifest", str(manifest), "--unity", str(unity),
                "--output-dir", str(output), "--init-ppo-checkpoint", str(checkpoint),
                "--device", "cpu", "--num-envs", "1", "--total-updates", "1",
                "--rollout-steps", "2", "--bptt-len", "1", "--chunks-per-minibatch", "2",
                "--ppo-epochs", "1", "--critic-warmup-updates", "1", "--training-safety-shield",
            ]
            with patch.object(sys, "argv", argv):
                args = parse_args()
            with patch.object(trainer, "UnityPointGoalEnv", ShieldedEnv), patch.object(
                trainer, "build_pointgoal_actor_critic", side_effect=model_factory,
            ), patch.object(trainer, "Categorical", side_effect=distribution_factory), patch.object(
                trainer, "select_rollout_action",
                side_effect=lambda logits, _mode: torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device),
            ), patch.object(trainer.logging, "basicConfig"), patch.object(
                trainer.logging, "FileHandler", return_value=trainer.logging.NullHandler(),
            ):
                trainer.run_training(args)
            self.assertEqual(environments[0].proposals, [0, 0])
            self.assertTrue(environments[0].closed)
            self.assertEqual(histories[0][0, -1].item(), 3)
            self.assertEqual(histories[1][0, -1].item(), 1)
            self.assertEqual(histories[2][0, -1].item(), 1)
            self.assertTrue(likelihood_actions)
            self.assertTrue(all((actions == 0).all() for actions in likelihood_actions))
            record = json.loads((output / "metrics.jsonl").read_text())
            self.assertEqual(record["executed_action_rates"], [0.0, 1.0, 0.0])
            self.assertEqual(record["shield_intervention_fraction"], 1.0)
            self.assertEqual(record["teacher_action_fraction"], 0.0)
            self.assertIsNone(record["teacher_source"])

    def test_proposal_likelihood_gives_correct_gradient_through_fixed_shield(self):
        logits = torch.tensor([0.3, 0.7, -0.2], requires_grad=True)
        probabilities = logits.softmax(-1)
        log_probabilities = logits.log_softmax(-1)
        mapping = torch.tensor([1, 1, 2])
        rewards = torch.tensor([0.0, 2.0, -1.0])[mapping]
        exact = torch.autograd.grad((probabilities * rewards).sum(), logits, retain_graph=True)[0]
        proposal_gradient = torch.autograd.grad(
            (probabilities.detach() * rewards * log_probabilities).sum(), logits, retain_graph=True,
        )[0]
        wrong_gradient = torch.autograd.grad(
            (probabilities.detach() * rewards * log_probabilities[mapping]).sum(), logits,
        )[0]
        torch.testing.assert_close(proposal_gradient, exact)
        self.assertFalse(torch.allclose(wrong_gradient, exact))


if __name__ == "__main__":
    unittest.main()
