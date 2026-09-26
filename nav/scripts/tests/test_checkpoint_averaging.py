"""Safety and numerical contracts for offline single-policy averaging."""

import hashlib
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from nav.train.checkpoint_averaging import average_state_dicts
from nav.models.policies import DaggerTransformerActorCritic, dagger_ppo_config
from nav.scripts.rl.average_ppo_checkpoints import build_average


class CheckpointAveragingTest(unittest.TestCase):
    def test_uniform_mean_and_frozen_identity_without_mutation(self):
        a = {"head.weight": torch.tensor([1., 3.]), "depth_encoder.weight": torch.tensor([7.]),
             "counter": torch.tensor(2)}
        b = {**a, "head.weight": torch.tensor([3., 7.])}
        c = {**a, "head.weight": torch.tensor([5., 11.])}
        result = average_state_dicts([a, b, c], fixed_prefixes=("depth_encoder.",))
        torch.testing.assert_close(result["head.weight"], torch.tensor([3., 7.]), rtol=0, atol=0)
        self.assertTrue(torch.equal(result["depth_encoder.weight"], a["depth_encoder.weight"]))
        result["head.weight"].zero_()
        self.assertTrue(torch.equal(a["head.weight"], torch.tensor([1., 3.])))

    def test_identical_float_state_is_bitwise_preserved(self):
        a = {"value": torch.tensor([1e-20, 3.1415927, 1e20])}
        self.assertTrue(torch.equal(average_state_dicts([a, a, a])["value"], a["value"]))

    def test_frozen_encoder_and_integer_differences_rejected(self):
        for key, dtype in [("depth_encoder.weight", torch.float32), ("counter", torch.int64)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differs"):
                average_state_dicts([{key: torch.tensor(1, dtype=dtype)},
                                     {key: torch.tensor(2, dtype=dtype)}], fixed_prefixes=("depth_encoder.",))

    def test_nonfinite_rejected_even_when_identical(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            a = {"head.weight": torch.tensor([value])}
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Nonfinite"):
                average_state_dicts([a, a])

    def test_structure_and_dtype_mismatch_rejected(self):
        base = {"x": torch.ones(2)}
        for other in ({"y": torch.ones(2)}, {"x": torch.ones(3)}, {"x": torch.ones(2, dtype=torch.float64)}):
            with self.subTest(other=other), self.assertRaises(ValueError):
                average_state_dicts([base, other])
        for states in ([], [base], [{}, {}]):
            with self.subTest(states=states), self.assertRaises(ValueError):
                average_state_dicts(states)

    def test_large_finite_values_do_not_overflow(self):
        result = average_state_dicts([{"x": torch.tensor([3e38])}, {"x": torch.tensor([2e38])}])
        self.assertTrue(torch.isfinite(result["x"]).all())
        torch.testing.assert_close(result["x"], torch.tensor([2.5e38]))

    def test_real_checkpoint_build_load_reset_and_provenance(self):
        torch.set_num_threads(1)
        config = dagger_ppo_config(dict(
            policy_type="transformer", use_depth=True, use_rgb=False,
            navigation_only_actions=True, chunk_size=1,
            goal_encoding="unity_egocentric_v2", goal_rep="polar",
            goal_distance_scale_m=50.0, seq_len=6, num_layers=1,
            depth_backbone="resnet18", img_size=32, half_width=False,
        ))
        model = DaggerTransformerActorCritic(config)
        a = model.state_dict()
        b = {**a, "head.0.weight": a["head.0.weight"] + .01}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "a.pt", root / "b.pt"]
            for path, state in zip(paths, (a, b)):
                torch.save({"model": state, "model_config": asdict(config),
                            "optimizer": {"stale": True}, "update": 100, "global_steps": 1000}, path)
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            report = build_average(paths, hashes, root / "average")
            merged = torch.load(root / "average/averaged_actor.pt", map_location="cpu", weights_only=False)
            self.assertNotIn("optimizer", merged)
            self.assertNotIn("train_config", merged)
            self.assertEqual(merged["update"], 0)
            self.assertEqual(report["inference_models"], 1)
            self.assertEqual([p["sha256"] for p in report["parents"]], hashes)
            self.assertEqual(report["changed_tensors_from_first"], 1)
            model.load_state_dict(merged["model"], strict=True)
            with self.assertRaises(FileExistsError):
                build_average(paths, hashes, root / "average")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_average(paths, ["wrong", hashes[1]], root / "bad_hash")
            self.assertFalse((root / "bad_hash").exists())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                build_average([paths[0], paths[0]], [hashes[0], hashes[0]], root / "duplicate")
            self.assertFalse((root / "duplicate").exists())


if __name__ == "__main__":
    unittest.main()
