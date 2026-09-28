"""Optional clipping separates detached-critic updates without changing defaults."""

import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from nav.baselines.rl.trainer import run_training
from nav.scripts.rl.train_pointgoal_ppo import parse_args
from nav.train.optimizers import clip_policy_value_gradients, gradient_l2_norm


class PolicyValueClippingTest(unittest.TestCase):
    def model(self):
        model = nn.Module()
        model.actor = nn.Linear(2, 1)
        model.critic = nn.Linear(2, 1)
        model.critic_memory = nn.Linear(2, 1)
        model.config = SimpleNamespace(policy_architecture="dagger_transformer_memory")
        model.value_parameter_prefixes = ("critic.", "critic_memory.")
        return model

    def gradients(self, model, actor=0.1, value=10.):
        for name, p in model.named_parameters():
            p.grad = torch.full_like(p, value if name.startswith("critic") else actor)

    def test_default_matches_pytorch_gradients_norm_and_adam_update_exactly(self):
        torch.manual_seed(174)
        a = self.model(); b = copy.deepcopy(a)
        self.gradients(a); self.gradients(b)
        opt_a = torch.optim.AdamW(a.parameters(), lr=1e-4, eps=1e-5)
        opt_b = torch.optim.AdamW(b.parameters(), lr=1e-4, eps=1e-5)
        expected = torch.nn.utils.clip_grad_norm_(a.parameters(), .5)
        actual = clip_policy_value_gradients(b, .5)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        opt_a.step(); opt_b.step()
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_large_value_gradients_do_not_shrink_small_actor_gradients(self):
        model = self.model(); self.gradients(model)
        expected = gradient_l2_norm(model.parameters())
        actor = [p.grad.clone() for p in model.actor.parameters()]
        total = clip_policy_value_gradients(model, .5, mode="actor_value")
        torch.testing.assert_close(total, expected)
        for p, original in zip(model.actor.parameters(), actor):
            torch.testing.assert_close(p.grad, original, rtol=0, atol=0)
        self.assertLessEqual(gradient_l2_norm([*model.critic.parameters(), *model.critic_memory.parameters()]).item(), .5)

    def test_both_groups_are_clipped_and_combined_bound_is_not_global(self):
        model = self.model(); self.gradients(model, actor=10.)
        clip_policy_value_gradients(model, .5, mode="actor_value")
        self.assertAlmostEqual(gradient_l2_norm(model.actor.parameters()).item(), .5, places=6)
        self.assertAlmostEqual(gradient_l2_norm([*model.critic.parameters(), *model.critic_memory.parameters()]).item(), .5, places=6)
        self.assertGreater(gradient_l2_norm(model.parameters()).item(), .5)

    def test_warmup_missing_gradients_and_wrapped_model(self):
        model = self.model(); self.gradients(model)
        for p in model.actor.parameters(): p.grad = None
        wrapped = nn.Module(); wrapped.module = model
        clip_policy_value_gradients(wrapped, .5, mode="actor_value")
        self.assertTrue(all(p.grad is None for p in model.actor.parameters()))
        for p in model.parameters(): p.grad = None
        self.assertEqual(clip_policy_value_gradients(model, .5, mode="actor_value").item(), 0.)

    def test_invalid_inputs_reject_before_mutating_gradients(self):
        for mode, bound, arch in [("bad",.5,"dagger_transformer"),("actor_value",0.,"dagger_transformer"),("actor_value",float("nan"),"dagger_transformer"),("actor_value",.5,"gru")]:
            model=self.model(); model.config.policy_architecture=arch; self.gradients(model)
            before=[p.grad.clone() for p in model.parameters()]
            with self.assertRaises(ValueError): clip_policy_value_gradients(model,bound,mode=mode)
            for p,g in zip(model.parameters(),before): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        model=self.model(); self.gradients(model); model.critic.bias.grad.fill_(float('nan'))
        actor=[p.grad.clone() for p in model.actor.parameters()]
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            clip_policy_value_gradients(model,.5,mode='actor_value')
        for p,g in zip(model.actor.parameters(),actor): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)

    def test_cli_default_and_early_invalid_limit_guard(self):
        base=['train','--manifest','unused','--unity','unused','--output-dir','unused']
        with patch.object(sys,'argv',base): self.assertEqual(parse_args().gradient_clip_mode,'global')
        with patch.object(sys,'argv',base+['--gradient-clip-mode','actor_value','--max-grad-norm','0']): args=parse_args()
        with self.assertRaisesRegex(SystemExit,'finite positive max-grad-norm'): run_training(args)


if __name__ == "__main__":
    unittest.main()
