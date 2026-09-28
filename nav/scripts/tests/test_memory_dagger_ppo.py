"""Persistent memory must preserve initialization, reset cleanly, and learn via PPO."""

import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from nav.baselines.rl.agent import PPOPointGoalController
from nav.models.policies import (
    DaggerTransformerActorCritic, MemoryDaggerTransformerActorCritic, dagger_ppo_config,
)
from nav.train.initialization import add_persistent_pointgoal_memory


class MemoryDaggerPPOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(619)
        cls.config = dagger_ppo_config(dict(policy_type="transformer", use_depth=True,
            use_rgb=False, navigation_only_actions=True, chunk_size=1,
            goal_encoding="unity_egocentric_v2", goal_rep="polar", goal_distance_scale_m=50.,
            seq_len=6, num_layers=1, depth_backbone="resnet18", img_size=32,
            half_width=False, goal_action_residual=True))
        source = DaggerTransformerActorCritic(cls.config)
        with torch.no_grad():
            source.critic[-1].weight.fill_(.02)  # Also preserve a nonzero learned value head.
        cls.source = dict(model=source.state_dict(), model_config=asdict(cls.config), update=25, global_steps=76800)

    def model(self, size=32):
        model = MemoryDaggerTransformerActorCritic(replace(self.config,
            policy_architecture="dagger_transformer_memory", persistent_memory_size=size))
        model.load_base_policy(self.source["model"])
        return model

    def test_initializer_and_real_depth_controller_parity_with_resets_and_overrides(self):
        self._controller_parity(False)

    def test_motion_initializer_real_depth_parity_with_resets_and_executed_overrides(self):
        self._controller_parity(True)

    def _controller_parity(self, motion_features):
        payload, audit = add_persistent_pointgoal_memory(self.source, 32, motion_features=motion_features)
        self.assertEqual(audit["maximum_logit_error"], 0.)
        self.assertEqual(audit["maximum_value_error"], 0.)
        self.assertTrue(audit["original_weights_identical"])
        self.assertNotIn("optimizer", payload)
        with tempfile.TemporaryDirectory() as directory:
            base_path, new_path = Path(directory)/"base.pt", Path(directory)/"memory.pt"
            torch.save(self.source, base_path)
            torch.save(payload, new_path)
            old, new = PPOPointGoalController(str(base_path), "cpu"), PPOPointGoalController(str(new_path), "cpu")
            old_logits, new_logits = [], []
            old.model.register_forward_hook(lambda _m, _a, out: old_logits.append(out[0]))
            new.model.register_forward_hook(lambda _m, _a, out: new_logits.append(out[0]))
            rng = np.random.default_rng(441)
            for step in range(20):
                if step == 13:
                    old.reset()
                    new.reset()
                state = dict(depth_obs=rng.random((39, 53, 1), dtype=np.float32),
                    curr_world_x=step*.2, curr_world_z=-step*.3, curr_yaw_deg=step*22.5,
                    target_world_x=15., target_world_z=9.)
                self.assertEqual(old.predict_action(**state), new.predict_action(**state))
                torch.testing.assert_close(old_logits[-1], new_logits[-1], atol=0, rtol=0)
                executed = ("turn left", "forward", "turn right")[step % 3]
                old.observe_executed_action(executed)
                new.observe_executed_action(executed)

    def test_motion_initializer_is_an_exact_zero_column_extension_of_seeded_memory(self):
        torch.manual_seed(123)
        control, _ = add_persistent_pointgoal_memory(self.source, 32)
        torch.manual_seed(123)
        motion, _ = add_persistent_pointgoal_memory(self.source, 32, motion_features=True)
        for name, original in control["model"].items():
            actual = motion["model"][name]
            if name == "recurrent_memory.weight_ih":
                torch.testing.assert_close(actual[:, :original.shape[1]], original, atol=0, rtol=0)
                self.assertEqual(int(torch.count_nonzero(actual[:, original.shape[1]:])), 0)
            else:
                torch.testing.assert_close(actual, original, atol=0, rtol=0)

    def test_motion_inputs_replay_exactly_and_use_current_executed_action(self):
        model = MemoryDaggerTransformerActorCritic(replace(self.config,
            policy_architecture="dagger_transformer_memory", persistent_memory_size=32,
            memory_motion_features=True))
        model.load_base_policy(self.source["model"])
        captured = []
        model.recurrent_memory.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().clone()))
        visual = torch.randn(1, 256)
        history = torch.full((1, 6), 3)
        hidden = model.initial_hidden(1, "cpu")
        _, _, hidden, _ = model(visual, torch.tensor([[.2, 0, 1.]]), history, None, hidden, torch.zeros(1, 1))
        self.assertEqual(int(torch.count_nonzero(captured[-1][:, -6:])), 0)
        history[:, -1] = 0  # Executed forward, regardless of previous prediction.
        before = hidden.clone()
        goal = torch.tensor([[.185, 0, 1.]])
        rollout = model(visual, goal, history, None, hidden, torch.ones(1, 1))
        replay = model(visual, goal, history, None, before, torch.ones(1, 1))
        torch.testing.assert_close(captured[-1], captured[-2], atol=0, rtol=0)
        torch.testing.assert_close(captured[-1][:, -6:], torch.tensor([[1., 1, 0, 0, .75, .75]]), atol=1e-5, rtol=0)
        for old, new in zip(rollout, replay):
            torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_multirow_memory_packing_and_frozen_reference_window(self):
        model = self.model(300)
        hidden = model.initial_hidden(2, "cpu")
        self.assertEqual(tuple(hidden.shape), (2, 7, 259))
        logits, value, hidden, _ = model(torch.randn(2, 256), torch.tensor([[.2, 0, 1.]]*2),
            torch.full((2, 6), 3), None, hidden, torch.zeros(2, 1))
        self.assertEqual(tuple(model.reference_window_hidden(hidden).shape), (2, 5, 259))
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.isfinite(hidden).all())
        self.assertEqual(int(torch.count_nonzero(hidden[:, 5:].reshape(2, -1)[:, 300:])), 0)

    def test_frozen_base_still_allows_memory_and_value_learning(self):
        model = MemoryDaggerTransformerActorCritic(replace(self.config,
            policy_architecture="dagger_transformer_memory", persistent_memory_size=32,
            freeze_base_actor=True))
        model.load_base_policy(self.source["model"])
        for name, parameter in model.named_parameters():
            expected = name.startswith(model.memory_parameter_prefixes + ("critic.",))
            self.assertEqual(parameter.requires_grad, expected, name)

    def test_memory_backpropagates_across_steps_and_resets_but_critic_is_isolated(self):
        model = self.model()
        visual = torch.randn(2, 256)
        goal, actions = torch.tensor([[.2, .2, .98]]*2), torch.full((2, 6), 3)
        hidden = model.initial_hidden(2, "cpu")
        _, value, _, _ = model(visual, goal, actions, None, hidden, torch.zeros(2, 1))
        (value - 2.).square().mean().backward()
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters()
                            if not n.startswith(model.value_parameter_prefixes)))
        self.assertTrue(any(p.grad is not None for p in model.critic_memory.parameters()))
        with torch.no_grad():
            model.memory_actor.weight.normal_(std=.1)
        for reset in (False, True):
            model.zero_grad(set_to_none=True)
            initial = model.initial_hidden(2, "cpu").requires_grad_()
            hidden = initial
            for step in range(4):
                masks = torch.zeros(2, 1) if reset and step == 0 else torch.ones(2, 1)
                logits, _, hidden, _ = model(visual, goal, actions, None, hidden, masks)
            logits[:, 0].sum().backward()
            memory_grad = initial.grad[:, model.seq_len-1:].abs().sum().item()
            self.assertEqual(memory_grad == 0., reset)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.recurrent_memory.parameters()))
            self.assertTrue(all(p.grad is None for p in model.depth_encoder.parameters()))

    def test_optimizer_roles_keep_old_actor_lr_separate_from_new_memory(self):
        from nav.baselines.rl.trainer import optimizer_parameter_groups, is_value_parameter
        model = self.model()
        groups = optimizer_parameter_groups(model, lr=1e-4, encoder_lr_scale=.1,
                                            policy_lr_scale=.01, adaptation_lr_scale=.5)
        roles = {id(p): (g["group_name"], g["lr"]) for g in groups for p in g["params"]}
        self.assertEqual(len(roles), len(list(model.parameters())))
        self.assertEqual(sum(len(g["params"]) for g in groups), len(roles))
        for name, p in model.named_parameters():
            if name.startswith(model.adaptation_parameter_prefixes):
                self.assertEqual(roles[id(p)], ("policy_adaptation", 5e-5))
            elif is_value_parameter(model, name):
                self.assertEqual(roles[id(p)], ("value", 1e-4))
            elif not name.startswith("depth_encoder."):
                self.assertEqual(roles[id(p)][0], "policy_temporal")
                self.assertAlmostEqual(roles[id(p)][1], 1e-6)

    def test_full_bptt_ppo_with_frozen_reference_warmup_and_memory_updates(self):
        self._run_full_memory_ppo(1.)

    def test_tempered_bptt_ppo_preserves_warmup_and_learns_memory(self):
        self._run_full_memory_ppo(.5)

    def test_motion_bptt_ppo_preserves_frozen_base_and_learns_motion_columns(self):
        self._run_full_memory_ppo(1., motion_features=True)

    def test_separate_clipping_full_ppo_preserves_warmup_and_frozen_motion_base(self):
        self._run_full_memory_ppo(1., motion_features=True, clip_mode="actor_value")

    def test_value_losses_do_not_create_actor_or_memory_gradients(self):
        model = self.model()
        with torch.no_grad(): model.critic_memory.weight.fill_(.02)
        visual = torch.randn(2, model.config.visual_dim)
        goal = torch.tensor([[.3, 0., 1.], [.6, .3, .95]])
        actions = torch.zeros(2, model.config.action_history_len, dtype=torch.long)
        _, values, _, _ = model.forward_from_visual(visual, goal, actions, None,
            model.initial_hidden(2, "cpu"), torch.zeros(2, 1))
        values.square().sum().backward()
        for name, p in model.named_parameters():
            if not name.startswith(model.value_parameter_prefixes):
                self.assertIsNone(p.grad, name)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.critic.parameters()))

    def _run_full_memory_ppo(self, temperature, motion_features=False, clip_mode="global"):
        from nav.baselines.rl import trainer
        from nav.scripts.rl.train_pointgoal_ppo import parse_args
        class Env:
            def __init__(self, *_args, **_kwargs): pass
            def reset(self):
                return {"depth": np.full((1, 32, 32), .1, np.float32),
                        "goal": np.array([.3, 0., 1.], np.float32)}
            def step(self, action):
                return self.reset(), .2 if action == 0 else -.1, False, {
                    "executed_action_index": action, "shield_intervened": False, "map_progress_valid": True}
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, initial, manifest, unity = [root/name for name in ("reference.pt", "initial.pt", "tasks.jsonl", "unity")]
            torch.save(self.source, source)
            payload, _ = add_persistent_pointgoal_memory(self.source, 32,
                motion_features=motion_features, freeze_base_actor=motion_features)
            torch.save(payload, initial)
            manifest.write_text("".join(json.dumps(dict(scene_name=f"scene{s}", split="train"))+"\n" for s in (1, 2)))
            unity.touch()
            output = root / "training"
            argv = ["train", "--manifest", str(manifest), "--unity", str(unity),
                "--output-dir", str(output), "--init-ppo-checkpoint", str(initial),
                "--reference-policy-checkpoint", str(source), "--reference-kl-coef", ".1",
                "--navigation-map-dir", str(root), "--device", "cpu", "--num-envs", "2",
                "--total-updates", "2", "--rollout-steps", "4", "--bptt-len", "2",
                "--chunks-per-minibatch", "2", "--ppo-epochs", "2", "--critic-warmup-updates", "1",
                "--checkpoint-interval", "1", "--lr", ".001", "--policy-lr-scale", ".01",
                "--policy-temperature", str(temperature), "--gradient-clip-mode", clip_mode]
            with patch.object(sys, "argv", argv): args = parse_args()
            with patch.object(trainer, "UnityPointGoalEnv", Env), patch.object(trainer.logging, "basicConfig"), patch.object(
                    trainer.logging, "FileHandler", return_value=trainer.logging.NullHandler()):
                trainer.run_training(args)
            warm = torch.load(output/"update_000001.pt", map_location="cpu", weights_only=False)["model"]
            final = torch.load(output/"latest.pt", map_location="cpu", weights_only=False)["model"]
            for name, value in payload["model"].items():
                if not name.startswith(("critic.", "critic_memory.")):
                    torch.testing.assert_close(warm[name], value, atol=0, rtol=0)
                if name.startswith("depth_encoder."):
                    torch.testing.assert_close(final[name], value, atol=0, rtol=0)
                if motion_features and not name.startswith(MemoryDaggerTransformerActorCritic.memory_parameter_prefixes + ("critic.",)):
                    torch.testing.assert_close(final[name], value, atol=0, rtol=0)
            self.assertGreater(float(final["memory_actor.weight"].abs().sum()), 0)
            self.assertFalse(torch.equal(final["recurrent_memory.weight_ih"], payload["model"]["recurrent_memory.weight_ih"]))
            if motion_features:
                self.assertGreater(float(final["recurrent_memory.weight_ih"][:, -6:].abs().sum()), 0)
            rows = [json.loads(l) for l in (output/"metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["teacher_action_fraction"] == 0 and r["teacher_coefficient"] == 0 for r in rows))
            self.assertTrue(all(np.isfinite(r["reference_kl"]) for r in rows))
            self.assertTrue(all(r["policy_temperature"] == temperature for r in rows))
            self.assertTrue(all(r["gradient_clip_mode"] == clip_mode for r in rows))
            self.assertEqual(rows[0]["gradient_norm_policy_after_clip"], 0.)
            self.assertGreater(rows[-1]["gradient_norm_policy_after_clip"], 0.)
            self.assertTrue(all(r["gradient_norm_policy_after_clip"] <= args.max_grad_norm + 1e-6
                                and r["gradient_norm_value_after_clip"] <= args.max_grad_norm + 1e-6 for r in rows))


if __name__ == "__main__":
    unittest.main()
