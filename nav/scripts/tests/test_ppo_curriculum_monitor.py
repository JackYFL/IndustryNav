"""Exercise validation-first scheduling and target gates without Unity/GPU."""

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nav.scripts.rl import evaluate_ppo_curriculum as monitor
from nav.scripts.tests.test_ppo_result_audit import make_report


class CurriculumMonitorTest(unittest.TestCase):
    def run_monitor(self, scores, *, fail_update=None, changed_budget=None, pin_budgets=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        initial = root / "initial.pt"
        initial.write_bytes(b"initial checkpoint")
        benchmark = root / "baseline.json"
        benchmark.write_text(json.dumps(make_report(59)))
        canonical = root / "input_points.json"
        canonical.write_bytes(b"canonical test fixture")
        validation = root / "val.jsonl"
        validation.write_text("".join(json.dumps(dict(scene_name=f"scene{s}",
            episode_id=f"independent_val_{p}", split="val")) + "\n"
            for s in range(1, 25) for p in range(1, 5)))
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir()
        for update in scores:
            if update:
                (checkpoint_dir / f"update_{update:06d}.pt").write_bytes(str(update).encode())
        output = root / "evaluations"
        digest = hashlib.sha256
        def fixture_hash(data):
            if data == b"canonical test fixture":
                return SimpleNamespace(hexdigest=lambda:
                    "1cb2062f53a68832ec2815bd62a61d0100c5fdcb58882602dda44f05127face2")
            return digest(data)
        calls = []
        def evaluate(command, **_kwargs):
            directory = Path(command[command.index("--output-root") + 1])
            update = int(directory.parent.name.split("_")[-1])
            split = directory.name
            calls.append((update, split))
            if pin_budgets and not (update==0 and split=="validation"):
                reference=Path(command[command.index("--step-budget-reference")+1])
                expected=benchmark if split=="benchmark96" else output/"update_000000/validation/summary.json"
                self.assertEqual(reference,expected)
                self.assertEqual(command[command.index("--step-budget-reference-sha256")+1],
                                 digest(reference.read_bytes()).hexdigest())
            else:
                self.assertNotIn("--step-budget-reference",command)
            if update == fail_update:
                return SimpleNamespace(returncode=1)
            result = copy.deepcopy(make_report(scores[update][split]))
            if split == "validation":
                for row in result["episodes"]:
                    row["episode_id"] = "independent_val_" + row["episode_id"][5:]
            if update == changed_budget:
                result["episodes"][0]["step_budget"] = 81
            (directory / "summary.json").write_text(json.dumps(result))
            return SimpleNamespace(returncode=0)
        argv = ["monitor", "--checkpoint-dir", str(checkpoint_dir),
            "--initial-checkpoint", str(initial), "--initial-benchmark-summary", str(benchmark),
            "--initial-benchmark-checkpoint-sha256", digest(initial.read_bytes()).hexdigest(),
            "--input-points", str(canonical), "--validation-manifest", str(validation),
            "--unity", "unused", "--output-root", str(output),
            "--milestones", *map(str, sorted(u for u in scores if u)),
            "--benchmark-milestones", "100"]
        if pin_budgets:argv.append("--pin-initial-budgets")
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch.object(monitor.hashlib, "sha256", side_effect=fixture_hash))
            stack.enter_context(patch.object(monitor.subprocess, "run", side_effect=evaluate))
            stack.enter_context(redirect_stdout(StringIO()))
            monitor.main()
            state = json.loads((output / "monitor.json").read_text())
            count = len(calls)
            # Completed/candidate milestones are not re-evaluated on resume.
            monitor.main()
            self.assertEqual(len(calls), count)
        return state, calls, output

    def test_validation_first_skip_and_forced_benchmark_candidate(self):
        state, calls, output = self.run_monitor({
            0: {"validation": 80}, 25: {"validation": 79},
            50: {"validation": 81, "benchmark96": 60},
            100: {"validation": 80, "benchmark96": 77},
        })
        self.assertEqual(calls, [(0, "validation"), (25, "validation"),
            (50, "validation"), (50, "benchmark96"),
            (100, "validation"), (100, "benchmark96")])
        self.assertTrue(state["target_candidate"])
        self.assertTrue((output / "target_candidate.json").is_file())
        self.assertEqual(state["evaluations"][0]["benchmark96"]["successes"], 59)

    def test_pinned_budgets_bind_exact_initial_reports_and_survive_resume(self):
        state,calls,_=self.run_monitor({0:{"validation":80},
            25:{"validation":82,"benchmark96":61},100:{"validation":83,"benchmark96":77}},pin_budgets=True)
        self.assertTrue(state["identity"]["pin_initial_budgets"])
        self.assertEqual(len(state["identity"]["initial_benchmark_summary_sha256"]),64)
        self.assertTrue(state["target_candidate"])
        self.assertEqual(calls,[(0,"validation"),(25,"validation"),(25,"benchmark96"),
                               (100,"validation"),(100,"benchmark96")])

    def test_high_benchmark_does_not_override_validation_regression(self):
        state, _, output = self.run_monitor({0: {"validation": 80},
            100: {"validation": 79, "benchmark96": 90}})
        self.assertFalse(state["target_candidate"])
        self.assertFalse((output / "target_candidate.json").exists())

    def test_failed_evaluation_cannot_become_a_result(self):
        with self.assertRaisesRegex(RuntimeError, "Evaluation failed"):
            self.run_monitor({0: {"validation": 80}, 25: {"validation": 96}}, fail_update=25)

    def test_validation_budget_must_match_initialization(self):
        with self.assertRaisesRegex(RuntimeError, "budgets differ"):
            self.run_monitor({0: {"validation": 80}, 25: {"validation": 96}}, changed_budget=25)


if __name__ == "__main__":
    unittest.main()
