"""Rate-only optimizer adaptation preserves moments and explicit defaults."""

import copy
import sys
import unittest
from unittest.mock import patch

import torch

from nav.baselines.rl.trainer import run_training
from nav.scripts.rl.train_pointgoal_ppo import parse_args
from nav.train.optimizers import gradient_l2_norm, restore_optimizer_state


class TrainingOptimizerResumeTest(unittest.TestCase):
    def make_optimizer(self, rate):
        a = torch.nn.Parameter(torch.tensor([.3, -.5]))
        b = torch.nn.Parameter(torch.tensor([.2]))
        opt = torch.optim.AdamW([
            {"params": [a], "group_name": "policy", "lr": rate},
            {"params": [b], "group_name": "value", "lr": rate * 10},
        ], eps=1e-5, weight_decay=.02)
        return opt, (a, b)

    def saved_state(self):
        opt, params = self.make_optimizer(1e-4)
        sum(p.square().sum() for p in params).backward()
        opt.step()
        return copy.deepcopy(opt.state_dict())

    def test_default_is_normal_checkpoint_restore(self):
        state = self.saved_state()
        opt, _ = self.make_optimizer(5e-4)
        restore_optimizer_state(opt, state)
        self.assertEqual([g["lr"] for g in opt.param_groups], [1e-4, 1e-3])
        actual = opt.state_dict()
        for key in state["state"]:
            for field, value in state["state"][key].items():
                torch.testing.assert_close(actual["state"][key][field], value, atol=0, rtol=0)

    def test_override_changes_only_rates_not_moments_or_saved_settings(self):
        state = self.saved_state()
        original = copy.deepcopy(state)
        opt, _ = self.make_optimizer(5e-4)
        opt.param_groups[0]["weight_decay"] = .8
        restore_optimizer_state(opt, state, use_config_lrs=True)
        actual = opt.state_dict()
        self.assertEqual([g["lr"] for g in opt.param_groups], [5e-4, 5e-3])
        for group, saved in zip(actual["param_groups"], original["param_groups"]):
            self.assertEqual({k:v for k,v in group.items() if k != "lr"},
                             {k:v for k,v in saved.items() if k != "lr"})
        for key in original["state"]:
            for field, value in original["state"][key].items():
                torch.testing.assert_close(actual["state"][key][field], value, atol=0, rtol=0)
        self.assertEqual(state["param_groups"], original["param_groups"])

    def test_group_mismatch_fails_before_loading(self):
        for mutate in (lambda s:s["param_groups"][0].update(group_name="different"),
                       lambda s:s["param_groups"][0]["params"].append(99),
                       lambda s:s["param_groups"].pop()):
            state = self.saved_state();mutate(state)
            opt, _ = self.make_optimizer(5e-4)
            with self.assertRaises(ValueError):
                restore_optimizer_state(opt, state, use_config_lrs=True)
            self.assertFalse(opt.state)
            self.assertEqual(opt.param_groups[0]["lr"], 5e-4)

    def test_invalid_configured_rates_fail_before_loading(self):
        for rate in (0., -1., float("nan"), float("inf")):
            opt, _ = self.make_optimizer(5e-4)
            opt.param_groups[0]["lr"] = rate
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                restore_optimizer_state(opt, self.saved_state(), use_config_lrs=True)
            self.assertFalse(opt.state)

    def test_gradient_diagnostics_do_not_change_update(self):
        results = []
        for diagnostics in (False, True):
            opt, params = self.make_optimizer(1e-4)
            sum(p.square().sum() for p in params).backward()
            old = [p.grad.clone() for p in params]
            if diagnostics:
                expected = torch.cat(old).norm(2)
                torch.testing.assert_close(gradient_l2_norm(params), expected)
                for p, grad in zip(params, old):
                    torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)
            torch.nn.utils.clip_grad_norm_(params, .5)
            opt.step()
            results.append([p.detach().clone() for p in params])
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_absent_gradients_are_zero(self):
        _, params = self.make_optimizer(1e-4)
        self.assertEqual(gradient_l2_norm(params).item(), 0)
        self.assertEqual(gradient_l2_norm([]).item(), 0)

    def test_cli_default_and_invalid_override_fail_before_unity(self):
        base = ["train", "--manifest", "unused", "--unity", "unused", "--output-dir", "unused"]
        with patch.object(sys, "argv", base):
            self.assertEqual(parse_args().resume_optimizer_lr_mode, "checkpoint")
        for extras in ([], ["--resume"], ["--resume", "--no-resume-optimizer"]):
            with patch.object(sys, "argv", base + extras + ["--resume-optimizer-lr-mode", "config"]):
                args = parse_args()
            with self.assertRaisesRegex(SystemExit, "Configured-rate optimizer resume"):
                run_training(args)


if __name__ == "__main__":
    unittest.main()
