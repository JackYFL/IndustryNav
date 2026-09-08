"""Offline recovery tests: no paid API, external CLI or Unity executable."""

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nav.config import ACTIONS_CSV_FIELDS
from nav.harness.checkpoint import CheckpointStore, RunLock, validate_restored_pose
from nav.harness.navigation_protocol import navigation_run_config
from nav.scripts.agent import run_benchmark_cell as runner
from nav.scripts.agent.resume_benchmark import resume_command, resume_environment


def empty_state():
    return {"phase": "ready", "step_count": 0, "step_budget": 3, "initial_step_budget": 3,
            "step_budget_mode": "fixed", "pose": {"x": 0., "y": .5, "z": 0., "rx": 0., "yaw": 180., "rz": 0.},
            "init_world": [0., 0.], "target_world": [10., -10.], "history": [],
            "pending_decision": None, "resume_count": 0}


class CheckpointStorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"OPENAI_MAX_REQUESTS": "", "OPENROUTER_MAX_REQUESTS": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        config = navigation_run_config(argparse.Namespace(llm_provider="openai"), "prompt")
        self.store = CheckpointStore(self.root, config, self.root / "llm_actions.csv", self.root / "agent_qa.txt")
        with (self.root / "llm_actions.csv").open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=ACTIONS_CSV_FIELDS).writeheader()
        (self.root / "agent_qa.txt").write_text("")

    def test_roundtrip_and_modification_guard(self):
        self.store.save(empty_state())
        self.assertEqual(self.store.load()["step_count"], 0)
        with (self.root / "llm_actions.csv").open("r+b") as stream:
            stream.write(b"bad")
        with self.assertRaisesRegex(ValueError, "modified or truncated"):
            self.store.load()

    def test_uncommitted_tails_are_archived_not_silently_deleted(self):
        state = self.store.save(empty_state())
        with (self.root / "llm_actions.csv").open("a") as stream:
            stream.write("uncommitted tail")
        (self.root / "llm_fp").mkdir()
        (self.root / "llm_fp/0.png").write_bytes(b"old frame")
        (self.root / "results.csv").write_text("old partial summary")
        archive = self.store.restore_logs(state)
        self.assertIn("uncommitted tail", (archive / "llm_actions.csv").read_text())
        self.assertEqual((archive / "llm_fp/0.png").read_bytes(), b"old frame")
        self.assertEqual((archive / "results.csv").read_text(), "old partial summary")
        self.assertEqual(self.store.load()["step_count"], 0)

    def test_reply_is_durable_before_the_main_loop_consumes_it(self):
        self.store.save(empty_state())
        self.store.save_reply(0, {"action": "forward", "reasoning": "clear", "observation": "clear floor",
                                  "prompt": "navigate", "history_entry": {}})
        restored = self.store.load()
        self.assertEqual(restored["phase"], "decision_ready")
        self.assertEqual(restored["pending_decision"]["action"], "forward")

    def test_wrong_step_reply_is_ignored(self):
        self.store.save(empty_state())
        self.store.save_reply(9, {"action": "forward", "reasoning": "clear", "observation": "clear",
                                  "prompt": "navigate", "history_entry": {}})
        self.assertEqual(self.store.load()["phase"], "ready")

    def test_invalid_pose_budget_and_steps_are_rejected(self):
        for changes in ({"step_count": -1}, {"step_count": 4}, {"step_budget": -1},
                        {"pose": {**empty_state()["pose"], "x": float("nan")}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.store.save({**empty_state(), **changes})

    def test_wrong_config_and_missing_counter_fail_closed(self):
        counter = self.root / "budget.sqlite3"
        with sqlite3.connect(counter) as conn:
            conn.execute("CREATE TABLE request_counter (id INTEGER PRIMARY KEY, value INTEGER)")
            conn.execute("INSERT INTO request_counter VALUES (1,5)")
        with patch.dict(os.environ, {"OPENAI_MAX_REQUESTS": "10", "OPENAI_REQUEST_COUNTER_FILE": str(counter)}):
            self.store.save(empty_state())
            self.assertEqual(self.store.load()["request_budget"]["requests_reserved"], 5)
            with patch.dict(os.environ, {"OPENAI_MAX_REQUESTS": "11"}), self.assertRaisesRegex(ValueError, "allowance"):
                self.store.load()
            with sqlite3.connect(counter) as conn:
                conn.execute("UPDATE request_counter SET value=0")
            with self.assertRaisesRegex(ValueError, "allowance"):
                self.store.load()
            counter.unlink()
            with self.assertRaisesRegex(ValueError, "allowance"):
                self.store.load()
        self.store.config["settings"]["model_id"] = "different"
        with self.assertRaisesRegex(ValueError, "different navigation"):
            self.store.load()

    def test_lock_prevents_two_writers(self):
        first = RunLock(self.root)
        try:
            with self.assertRaisesRegex(ValueError, "Another process"):
                RunLock(self.root)
        finally:
            first.close()
        RunLock(self.root).close()

    def test_restore_validation_handles_yaw_wrap_and_rejects_wrong_spawn(self):
        expected = {**empty_state()["pose"], "yaw": 359.8}
        validate_restored_pose(expected, {**expected, "yaw": .2}, [10, -10], [10, -10])
        with self.assertRaisesRegex(ValueError, "did not restore"):
            validate_restored_pose(expected, {**expected, "x": 1}, [10, -10], [10, -10])

    def test_resume_helper_preserves_budget_and_never_prints_environment_keys(self):
        counter = self.root / "budget.sqlite3"
        counter.touch()
        env = resume_environment(self.store.config, {"request_budget": {"counter_file": str(counter), "limit": 7200}},
                                 {"OPENAI_API_KEY": "not-a-real-key", "OPENAI_MAX_REQUESTS": "5000"})
        self.assertEqual(env["OPENAI_MAX_REQUESTS"], "5000")
        self.assertEqual(env["OPENAI_REQUEST_COUNTER_FILE"], str(counter))
        with patch("nav.scripts.agent.resume_benchmark.free_tcp_port", return_value=5555):
            command = resume_command(self.root, self.store.config)
        self.assertNotIn("not-a-real-key", " ".join(command))
        self.assertIn("--resume", command)


class ImmediateThread:
    def __init__(self, target, args, **kwargs):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)

    def is_alive(self):
        return False

    def join(self):
        pass


class FakeEnv:
    def __init__(self, args, mode, calls):
        self.pose = [args.init_world_x, getattr(args, "_resume_world_y", .5), args.init_world_z, 0., args.init_curr_direction, 0.]
        self.mode, self.calls = mode, calls
        self.signal = [0., 0., 0.]
        self.action_ticks = 0
        self.closed = False

    def get_steps(self, behavior):
        return argparse.Namespace(obs=[
            np.ones((1, 3, 240, 320), dtype=np.float32) * .5,
            np.ones((1, 3, 256, 431), dtype=np.float32) * .8,
            np.ones((1, 1, 240, 320), dtype=np.float32) * .5,
            np.array([self.pose], dtype=np.float32),
        ]), None

    def set_actions(self, behavior, actions):
        self.signal = actions.continuous[0]

    def step(self):
        move, _, look = self.signal
        if abs(move) + abs(look) > 0:
            self.pose[0] += float(move) * .05 * math.sin(math.radians(self.pose[4]))
            self.pose[2] += float(move) * .05 * math.cos(math.radians(self.pose[4]))
            self.pose[4] = (self.pose[4] + float(look)) % 360
            self.action_ticks += 1
            if self.mode == "mid_action" and self.action_ticks == 3:
                raise RuntimeError("simulated Unity failure in the middle of an action")
        elif self.mode == "after_reply" and len(self.calls) == 2:
            raise RuntimeError("simulated Unity failure before main consumes reply")

    def close(self):
        self.closed = True


class CheckpointLoopTest(unittest.TestCase):
    def test_interrupted_runs_continue_without_rebilling_saved_decisions(self):
        for mode in ("mid_action", "after_reply", "legacy", "dynamic"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                calls, environments = [], []
                failure_mode = "mid_action" if mode in {"legacy", "dynamic"} else mode
                fail = [failure_mode]
                logger = Mock()
                expected_steps = 4 if mode == "dynamic" else 3
                target = (0., -3.1) if mode == "dynamic" else (10., -10.)

                def setup(args, logger):
                    env = FakeEnv(args, fail[0], calls)
                    environments.append(env)
                    return argparse.Namespace(env=env, init_world=(args.init_world_x, args.init_world_z),
                        target_world=target, target_xy=(200, 200), margin=(0, 431, 0, 256),
                        target_sc=argparse.Namespace(last_spawn_pixel=(100, 100), last_spawn_world=(0., 0.),
                                                   last_target_pixel=(200, 200), last_target_world=target))

                def decide(baseline, payload, result):
                    action = ("forward", "turn right", "forward", "forward")[len(calls)]
                    calls.append(payload)
                    result.update(action=action, error=False, finished=True, observation="Clear floor.",
                                  reasoning="Navigate safely.", prompt=payload["prompt"], history_entry=payload["history_entry"])

                argv = ["cell", "--scene_id", "0", "--scene_name", "scene1", "--point_id", "point1", "--seed_id", "0",
                        "--frame_save_dir", str(folder), "--model_id", "test-model", "--vision_input", "true",
                        "--init_world_x", "0", "--init_world_z", "0", "--init_curr_direction", "180",
                        "--target_x", "200", "--target_y", "200", "--max_steps", "3", "--no-dynamic_step_budget"]
                if mode == "dynamic":
                    argv += ["--dynamic_step_budget", "--step_budget_min", "1", "--step_budget_max", "10",
                             "--steps_per_path_meter", ".8", "--step_budget_overhead", "1", "--reach_m", ".1"]
                with patch.object(runner, "setup_and_prime", side_effect=setup), \
                     patch.object(runner, "execute_decision", side_effect=decide), \
                     patch.object(runner.threading, "Thread", ImmediateThread), \
                     patch.object(runner, "logger_config", return_value=logger), \
                     patch.object(runner, "resolve_scene_all_path", return_value="fake-unity"), \
                     patch.dict(os.environ, {"OPENROUTER_MAX_REQUESTS": ""}):
                    with patch("sys.argv", argv):
                        runner.main()
                    self.assertEqual(len(calls), 2, str(logger.exception.call_args))
                    config = json.loads((folder / "run_config.json").read_text())
                    store = CheckpointStore(folder, config, folder / "llm_actions.csv", folder / "agent_qa.txt")
                    committed_frame = (folder / "llm_fp/0.png").read_bytes()
                    if mode == "legacy":
                        (folder / "checkpoint.json").unlink()
                        (folder / "decision_reply.json").unlink()
                    else:
                        checkpoint = store.load()
                        self.assertEqual(checkpoint["step_count"], 1)
                        self.assertEqual(checkpoint["phase"], "decision_ready")
                    fail[0] = None
                    with patch("sys.argv", argv + ["--resume"]):
                        runner.main()
                    self.assertEqual(len(calls), expected_steps, "Saved second decision must not make another model call")
                    self.assertIn("Saw: Clear floor.", calls[-1]["prompt"])
                    checkpoint = store.load()
                    self.assertEqual(checkpoint["phase"], "completed")
                    self.assertEqual(checkpoint["step_count"], expected_steps)
                    self.assertEqual(checkpoint["initial_step_budget"], expected_steps)
                    self.assertEqual(checkpoint["resume_count"], 1)
                    with (folder / "llm_actions.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual([int(row["step"]) for row in rows], list(range(1, expected_steps + 1)))
                    self.assertEqual([row["action"] for row in rows], ["forward", "turn right", "forward", "forward"][:expected_steps])
                    self.assertEqual((folder / "llm_fp/0.png").read_bytes(), committed_frame)
                    self.assertEqual(len(list(folder.glob("resume-*/results.csv"))), 1)
                    with (folder / "results.csv").open() as stream:
                        summary = list(csv.DictReader(stream))[-1]
                    self.assertEqual(summary["steps_taken"], str(expected_steps))
                    self.assertEqual(summary["resume_count"], "1")
                    self.assertAlmostEqual(float(summary["final_world_x"]), checkpoint["pose"]["x"])
                    with patch("sys.argv", argv + ["--resume"]):
                        runner.main()
                    self.assertEqual(len(environments), 2, "Completed run must not launch Unity again")
                    self.assertTrue(all(env.closed for env in environments))
                    with patch("sys.argv", argv), self.assertRaisesRegex(SystemExit, "would be overwritten"):
                        runner.main()


if __name__ == "__main__":
    unittest.main()
