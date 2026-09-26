"""Reject incomplete, relaxed, or internally inconsistent benchmark reports."""

import copy
import unittest

from nav.scripts.rl.audit_pointgoal_ppo_result import PROTOCOL, audit_result, validate_report


def make_report(successes=57):
    episodes = []
    for i in range(96):
        success = i < successes
        episodes.append(dict(
            scene_name=f"scene{i // 4 + 1}", episode_id=f"point{i % 4 + 1}",
            returncode=0, success=success, distance_world=1.9 if success else 10.0,
            steps_taken=80, step_budget=80,
            stop_reason="reached_vicinity" if success else "max_steps",
            terminal_homing_interventions=0, fallback_activated=False,
            second_fallback_activated=False, collision_steps=0, collision_events=0,
            warning_steps=0, safety_shield_interventions=0,
            predicted_action_counts={"forward": 80}, executed_action_counts={"forward": 80},
        ))
    return dict(PROTOCOL, episodes=episodes, successes=successes, success_rate=successes / 96)


class PPOResultAuditTest(unittest.TestCase):
    def test_eighty_percent_requires_seventy_seven(self):
        self.assertFalse(audit_result(make_report(76),make_report())["score_above_80_percent"])
        self.assertTrue(audit_result(make_report(77),make_report())["score_above_80_percent"])

    def test_independent_validation_uses_explicit_task_keys(self):
        report=make_report()
        for row in report["episodes"]:
            row["episode_id"]="independent_val_"+row["episode_id"]
        keys={(r["scene_name"],r["episode_id"]) for r in report["episodes"]}
        self.assertEqual(len(validate_report(report,expected_keys=keys)),96)
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_strict_target_and_gains(self):
        baseline = make_report()
        self.assertFalse(audit_result(baseline, baseline)["score_above_60_percent"])
        result = audit_result(make_report(58), baseline)
        self.assertTrue(result["score_above_60_percent"])
        self.assertEqual(result["gained_tasks"], [["scene15", "point2"]])
        self.assertEqual(result["lost_tasks"], [])

    def test_reject_wrong_or_duplicate_tasks(self):
        for scene in ("scene25", "scene2"):
            with self.subTest(scene=scene):
                report = make_report()
                report["episodes"][0]["scene_name"] = scene
                with self.assertRaises(ValueError):
                    audit_result(report, make_report())

    def test_reject_relaxed_protocol_and_missing_fields(self):
        for field in PROTOCOL:
            with self.subTest(field=field):
                report = make_report()
                del report[field]
                with self.assertRaises(ValueError):
                    audit_result(report, make_report())
        report = make_report()
        report["reach_m"] = 5.0
        with self.assertRaises(ValueError):
            audit_result(report, make_report())

    def test_reject_bad_episode_evidence(self):
        changes = [
            ("returncode", 1), ("distance_world", float("nan")),
            ("distance_world", 2.1), ("success", "true"),
            ("steps_taken", 81), ("step_budget", 81),
            ("stop_reason", "interrupted"), ("collision_steps", -1),
            ("terminal_homing_interventions", 1), ("fallback_activated", True),
            ("executed_action_counts", {"forward": 79}),
        ]
        for field, value in changes:
            with self.subTest(field=field):
                report = make_report()
                report["episodes"][0][field] = value
                with self.assertRaises(ValueError):
                    audit_result(report, make_report())

    def test_reject_partial_and_misreported_score(self):
        baseline = make_report()
        for field, value in (("episodes", baseline["episodes"][:-1]), ("successes", 58), ("success_rate", 0.9)):
            with self.subTest(field=field):
                report = copy.deepcopy(baseline)
                report[field] = value
                with self.assertRaises(ValueError):
                    audit_result(report, baseline)


if __name__ == "__main__":
    unittest.main()
