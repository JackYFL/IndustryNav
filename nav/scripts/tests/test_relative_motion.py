"""Relative motion cues preserve PointGoal boundaries, resets and replay."""

import math
import unittest

import torch

from nav.models.policies.relative_motion import relative_motion_features


class RelativeMotionTest(unittest.TestCase):
    def features(self, before, after, actions, masks=None, rep="cartesian", scale=50.):
        before, after = torch.tensor(before, dtype=torch.float64), torch.tensor(after, dtype=torch.float64)
        return relative_motion_features(before, after, torch.tensor(actions),
            torch.ones(len(actions), 1) if masks is None else torch.tensor(masks),
            goal_rep=rep, distance_scale_m=scale)

    def test_free_blocked_sideways_and_away_forward_steps(self):
        result = self.features([[.2, 0], [.2, 0], [.2, .1], [-.2, 0]],
            [[.185, 0], [.2, 0], [.2, .085], [-.215, 0]], [0]*4)
        torch.testing.assert_close(result[:, 5], torch.tensor([.75, 0, .75, .75], dtype=torch.float64))
        self.assertAlmostEqual(result[0, 4].item(), .75)
        self.assertAlmostEqual(result[1, 4].item(), 0.)
        self.assertAlmostEqual(result[3, 4].item(), -.75)
        torch.testing.assert_close(result[:, :4], torch.tensor([[1., 1., 0, 0]]*4, dtype=torch.float64))

    def test_rotation_is_not_forward_displacement_and_progress_is_invariant(self):
        result = self.features([[.2, 0], [.2, .1]], [[.2, .125], [.2, -.025]], [1, 2], rep="polar")
        torch.testing.assert_close(result[:, 4:], torch.zeros(2, 2, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(result[:, 1:4], torch.tensor([[0., 1., 0.], [0., 0., 1.]], dtype=torch.float64))

    def test_reset_and_bos_are_zero_even_with_stale_nan_goals(self):
        result = self.features([[float("nan"), 0]]*3, [[.2, 0]]*3, [0, 3, 2], [[0], [1], [0]])
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(int(torch.count_nonzero(result)), 0)

    def test_polar_cartesian_equivalence_wrap_and_metres_not_goal_scale(self):
        before = torch.tensor([[-10., .01], [5., 4.]], dtype=torch.float64)
        after = torch.tensor([[-10.75, -.01], [4.25, 4.]], dtype=torch.float64)
        def polar(x, scale):
            return torch.stack((x.norm(dim=-1)/scale, torch.atan2(x[:, 1], x[:, 0])/math.pi), -1).tolist()
        expected = self.features((before/50).tolist(), (after/50).tolist(), [0, 0])
        for scale in (10., 50., 100.):
            actual = self.features(polar(before, scale), polar(after, scale), [0, 0], rep="polar", scale=scale)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_large_delta_is_bounded_and_inputs_unchanged(self):
        before, after = torch.tensor([[10., 0.]]), torch.tensor([[0., 0.]])
        saved = before.clone()
        result = relative_motion_features(before, after, torch.tensor([0]), torch.ones(1, 1),
            goal_rep="cartesian", distance_scale_m=50.)
        torch.testing.assert_close(result[:, 4:], torch.tensor([[2., 2.]]))
        torch.testing.assert_close(before, saved, atol=0, rtol=0)
        for rep, scale in (("invalid", 50.), ("polar", 0.), ("polar", float("nan"))):
            with self.assertRaises(ValueError):
                relative_motion_features(before, after, torch.tensor([0]), torch.ones(1, 1),
                                         goal_rep=rep, distance_scale_m=scale)


if __name__ == "__main__":
    unittest.main()
