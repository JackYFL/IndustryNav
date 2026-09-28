"""Validation reuse is tied to exact cohort bytes and original physics hashes."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nav.scripts.tools.finalize_independent_pointgoal_dataset import main


class IndependentFinalizationTest(unittest.TestCase):
    def finalize(self, *, change_val=False, change_physics=False, invalid_number=False,
                 legacy_validation=False, legacy_training=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        source, physics, previous, old_physics = [root / name for name in ("source", "physics", "prior", "prior_physics")]
        for path in (source, physics, previous, old_physics):
            path.mkdir()
        task = dict(scene_name="scene1", episode_id="long_train_0000", split="train")
        val = dict(scene_name="scene1", episode_id="independent_val_0000", split="val")
        (source / "train_manifest.jsonl").write_text(json.dumps(task) + "\n")
        for name in ("val_manifest.jsonl", "val_quick_manifest.jsonl"):
            (source / name).write_text(json.dumps(val) + "\n")
            (previous / name).write_bytes((source / name).read_bytes())
        (source / "sampling_audit.json").write_text("{}\n")
        checked = dict(status="checked", finite_depth=True, spawn_error_m=.01,
            target_error_m=.02, collision=True, map_progress_valid=False)
        new_check = dict(task, **checked)
        if invalid_number:
            new_check["spawn_error_m"] = float("nan")
        if legacy_training:
            del new_check["status"]
        (physics / "scene1.jsonl").write_text(json.dumps(new_check) + "\n")
        evidence = old_physics / "scene1.jsonl"
        old_check = dict(val, **checked)
        if legacy_validation:
            del old_check["status"]
        evidence.write_text(json.dumps(old_check) + "\n" +
            json.dumps(dict(task, episode_id="old_train_0000", **checked)) + "\n")
        (previous / "physical_audit.json").write_text(json.dumps({"physics_sha256": {
            str(evidence): hashlib.sha256(evidence.read_bytes()).hexdigest()}}))
        if change_val:
            (source / "val_manifest.jsonl").write_text(json.dumps(dict(val, changed=True)) + "\n")
        if change_physics:
            evidence.write_text(evidence.read_text() + "\n")
        output = root / "final"
        argv = ["finalize", "--source-dir", str(source), "--physics-dir", str(physics),
            "--output-dir", str(output), "--reuse-validation-dir", str(previous),
            "--validation-physics-dir", str(old_physics)]
        with patch("sys.argv", argv):
            main()
        return output, previous

    def test_keep_collision_cases_and_exact_reserved_validation(self):
        output, previous = self.finalize()
        audit = json.loads((output / "physical_audit.json").read_text())
        self.assertEqual((audit["training_pairs"], audit["validation_pairs"]), (1, 1))
        self.assertEqual(audit["physical_checked"], 2)
        self.assertEqual(audit["rejected"], [])
        for name in ("val_manifest.jsonl", "val_quick_manifest.jsonl"):
            # Manifest serialization is canonicalized by finalization.
            self.assertEqual([json.loads(l) for l in (output/name).read_text().splitlines()],
                             [json.loads(l) for l in (previous/name).read_text().splitlines()])

    def test_reject_changed_validation(self):
        with self.assertRaisesRegex(SystemExit, "byte-identical"):
            self.finalize(change_val=True)

    def test_reject_modified_prior_physical_evidence(self):
        with self.assertRaisesRegex(SystemExit, "hash mismatch"):
            self.finalize(change_physics=True)

    def test_reject_nonfinite_physical_errors(self):
        with self.assertRaisesRegex(SystemExit, "physical failure"):
            self.finalize(invalid_number=True)

    def test_legacy_success_schema_is_limited_to_verified_reused_validation(self):
        output, _ = self.finalize(legacy_validation=True)
        self.assertTrue((output / "physical_audit.json").exists())
        with self.assertRaisesRegex(SystemExit, "physical failure"):
            self.finalize(legacy_training=True)


if __name__ == "__main__":
    unittest.main()
