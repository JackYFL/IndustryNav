"""Regression checks for preserving a DAgger actor during PPO initialization."""

import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from nav.baselines.bc.agent import BCNavController
from nav.baselines.rl.agent import PPOPointGoalController
from nav.core.pointgoal import encode_rl_goal
from nav.models.policies import (
    DaggerTransformerActorCritic, NavPolicyTransformer, dagger_ppo_config,
)


class DaggerPPOActorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(8401)
        cls.config = dict(
            policy_type="transformer", use_depth=True, use_rgb=False,
            navigation_only_actions=True, include_stop_targets=False,
            chunk_size=1, goal_encoding="unity_egocentric_v2", goal_rep="polar",
            goal_distance_scale_m=50.0, seq_len=12, num_layers=3,
            depth_backbone="resnet18", img_size=32, half_width=False,
            goal_action_residual=True,
        )
        cls.source = NavPolicyTransformer(
            goal_dim=2, num_actions=3, use_depth=True, use_rgb=False,
            seq_len=12, num_layers=3, depth_backbone="resnet18", img_size=32,
            half_width=False, goal_action_residual=True,
            previous_action_vocab_size=4,
        ).eval()

    def test_controller_logits_match_full_dagger_actor_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dagger.pt"
            target = Path(directory) / "ppo.pt"
            torch.save({"model": self.source.state_dict(), "config": self.config}, source)
            cfg = dagger_ppo_config(self.config)
            model = DaggerTransformerActorCritic(cfg)
            model.load_dagger_actor(self.source.state_dict())
            torch.save({"model": model.state_dict(), "model_config": asdict(cfg)}, target)
            bc = BCNavController(str(source), device="cpu")
            ppo = PPOPointGoalController(str(target), device="cpu")
            original_logits, ppo_logits = [], []
            bc.model.register_forward_hook(lambda _m, _a, out: original_logits.append(out.detach()))
            ppo.model.register_forward_hook(lambda _m, _a, out: ppo_logits.append(out[0].detach()))
            rng = np.random.default_rng(4201)
            for step in range(20):
                if step == 15:
                    bc.reset()
                    ppo.reset()
                state = dict(
                    depth_obs=rng.random((39, 53, 1), dtype=np.float32),
                    curr_world_x=step * 0.37, curr_world_z=-step * 0.21,
                    curr_yaw_deg=step * 23.5, target_world_x=14., target_world_z=-20.,
                )
                expected = bc.predict_action(ego_obs=None, **state)
                actual = ppo.predict_action(**state)
                self.assertEqual(expected, actual)
                torch.testing.assert_close(original_logits[-1], ppo_logits[-1], atol=3e-5, rtol=3e-5)
                # Emulate external shield overrides, including a padded window.
                executed = ("forward", "turn right", "turn left")[step % 3]
                bc.observe_executed_action(executed)
                ppo.observe_executed_action(executed)

    def test_ppo_replay_likelihood_and_critic_gradient_isolation(self):
        model = DaggerTransformerActorCritic(dagger_ppo_config(self.config))
        model.load_dagger_actor(self.source.state_dict())
        goal = torch.tensor(encode_rl_goal(0, 0, 47, 6, 8)[None])
        depth = torch.rand(1, 1, 32, 32)
        actions = torch.full((1, 12), 3, dtype=torch.long)
        hidden = model.initial_hidden(1, torch.device("cpu"))
        masks = torch.zeros(1, 1)
        with torch.no_grad():
            old_logits, _, _, visual = model(depth, goal, actions, None, hidden, masks)
        model.train()
        logits, value, _, _ = model(visual, goal, actions, None, hidden, masks)
        torch.testing.assert_close(logits, old_logits, atol=1e-5, rtol=1e-5)
        (value - 2).square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.critic.parameters()))
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if not n.startswith("critic.")))
        model.zero_grad(set_to_none=True)
        logits, _, _, _ = model(visual, goal, actions, None, hidden, masks)
        (-logits.log_softmax(-1)[:, 1]).mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.depth_encoder.parameters()))

    def test_head_only_finetuning_preserves_actor_representation(self):
        model = DaggerTransformerActorCritic(replace(
            dagger_ppo_config(self.config), freeze_dagger_transformer=True,
        ))
        model.load_dagger_actor(self.source.state_dict())
        names = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(names)
        self.assertTrue(all(n.startswith(("head.", "critic.")) for n in names))
        goal = torch.tensor(encode_rl_goal(0, 0, 47, 6, 8)[None])
        depth = torch.rand(1, 1, 32, 32)
        actions = torch.full((1, 12), 3, dtype=torch.long)
        hidden = model.initial_hidden(1, torch.device("cpu"))
        logits, value, _, _ = model(depth, goal, actions, None, hidden, torch.zeros(1, 1))
        (-logits.log_softmax(-1)[:, 1].mean() + (value - 2).square().mean()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.goal_action_head.parameters()))


if __name__ == "__main__":
    unittest.main()
