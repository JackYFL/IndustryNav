"""Temperature must describe one consistent on-policy training distribution."""

import sys
import unittest
from unittest.mock import patch

import torch
from torch.distributions import Categorical, kl_divergence

from nav.baselines.rl.regularization import reference_kl, temperature_logits
from nav.baselines.rl.trainer import run_training
from nav.scripts.rl.train_pointgoal_ppo import parse_args


class PPOTemperatureTest(unittest.TestCase):
    def test_default_is_exact_identity_including_gradients(self):
        logits = torch.tensor([[.2, -.1, .7]], requires_grad=True)
        self.assertIs(temperature_logits(logits), logits)
        self.assertIs(temperature_logits(logits, 1.), logits)

    def test_sharpening_remains_stochastic_and_preserves_argmax(self):
        logits = torch.tensor([[.5, -.1, .7], [-.2, .8, .1]])
        base = Categorical(logits=logits)
        sharp = Categorical(logits=temperature_logits(logits, .5))
        self.assertTrue(torch.all(sharp.probs > 0))
        self.assertTrue(torch.all(sharp.entropy() < base.entropy()))
        torch.testing.assert_close(sharp.probs.argmax(-1), base.probs.argmax(-1))

    def test_replay_ratio_matches_behavior_and_unscaled_replay_does_not(self):
        logits = torch.tensor([[.2, -.1, .7], [.8, .1, -.2]])
        actions = torch.tensor([0, 2])
        old = Categorical(logits=temperature_logits(logits, .5)).log_prob(actions)
        new = Categorical(logits=temperature_logits(logits.clone(), .5)).log_prob(actions)
        torch.testing.assert_close((new-old).exp(), torch.ones(2), atol=0, rtol=0)
        wrong = Categorical(logits=logits).log_prob(actions)
        self.assertFalse(torch.allclose((wrong-old).exp(), torch.ones(2)))

    def test_reference_kl_uses_tempered_distributions_and_frozen_reference(self):
        logits = torch.tensor([[.5, -.2, .9]], requires_grad=True)
        reference = torch.tensor([[.1, .2, -.3]], requires_grad=True)
        a, b = temperature_logits(logits, .5), temperature_logits(reference, .5)
        actual = reference_kl(a, b)
        expected = kl_divergence(Categorical(logits=a), Categorical(logits=b.detach()))
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertIsNone(reference.grad)

    def test_invalid_temperatures_fail_before_environment_creation(self):
        for temperature in (0., -1., float("nan"), float("inf")):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    temperature_logits(torch.zeros(1,3), temperature)
                with patch.object(sys, "argv", ["train", "--manifest", "unused", "--unity", "unused",
                        "--output-dir", "unused", f"--policy-temperature={temperature}"]):
                    args = parse_args()
                with self.assertRaisesRegex(SystemExit, "policy-temperature"):
                    run_training(args)

    def test_cli_defaults_and_nondefault_teacher_guard(self):
        base = ["train", "--manifest", "unused", "--unity", "unused", "--output-dir", "unused"]
        with patch.object(sys, "argv", base):
            self.assertEqual(parse_args().policy_temperature, 1.)
        for extras in (["--online-astar-teacher"], ["--policy-loss-coef", "0"],
                       ["--teacher-action-prob", ".1"], ["--teacher-replay-capacity", "8"]):
            with self.subTest(extras=extras):
                with patch.object(sys, "argv", base + ["--policy-temperature", ".5"] + extras):
                    args = parse_args()
                with self.assertRaisesRegex(SystemExit, "pure stochastic PPO"):
                    run_training(args)


if __name__ == "__main__":
    unittest.main()
