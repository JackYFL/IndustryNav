"""Offline regressions for API/Kiro task parity (no network or Unity launch)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from collections import deque
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nav.config import ACTION_SPACE_AGENTS, DEFAULT_PROMPT_VISION, DEFAULT_PROMPT_NOVISION
from nav.harness.llm_provider import llm_generate_decision, parse_navigation_decision
from nav.harness.navigation_protocol import (
    check_navigation_run_config, navigation_run_config, prepare_navigation_run,
    resolve_navigation_sensors,
)
from nav.harness.prompt_assembly import (
    add_api_observation_contract, format_history_for_prompt, render_nav_prompt,
)
from nav.harness.routing import execute_decision
from nav.scripts import run_benchmark_cell as cell_runner
from nav.scripts import run_benchmark_grid as grid_runner
from nav.utils import load_prompt_template


def render_sample(history="No previous movements yet.", template=DEFAULT_PROMPT_VISION, **kwargs):
    return render_nav_prompt(
        load_prompt_template(template), (168, 184), (724, 478), 180, 0,
        list(ACTION_SPACE_AGENTS), history, curr_world_xz=(30.21, 54.69),
        target_world_xz=(7.25, 12.43), distance_m=48.09, **kwargs,
    )


class NavigationPromptTest(unittest.TestCase):
    def test_task_body_matches_archived_kiro_prompt_exactly(self):
        # SHA-256 of the navigation body, stripped of the transport-specific
        # wrapper, in scene1/point1/cli_agent_gpt-5.6-luna/seed0, Step 1.
        self.assertEqual(
            hashlib.sha256(render_sample().strip().encode()).hexdigest(),
            "f742a414829805f13619bd14e12e6b546e2ef692034e846cf54bdb43cfdab208",
        )

    def test_action_description_tracks_simulation_step_override(self):
        prompt = render_sample(sim_steps_per_decision=1)
        self.assertIn("yaw by about 22.5°", prompt)
        self.assertIn("moves about 0.75 m", prompt)
        self.assertIn("about 4 turn decision(s)", prompt)

    def test_wrapper_uses_inline_image_not_unavailable_file_tools(self):
        prompt = add_api_observation_contract(render_sample(), True)
        self.assertIn("attached image", prompt)
        self.assertNotIn("file-read tool", prompt)
        self.assertNotIn("/tmp/", prompt)
        self.assertIn('"observation"', prompt)

    def test_no_vision_contract_does_not_claim_an_image(self):
        prompt = add_api_observation_contract(render_sample(template=DEFAULT_PROMPT_NOVISION), False)
        self.assertIn("No camera or map images are attached.", prompt)
        self.assertIn("increasing world X is North", prompt)
        self.assertNotIn("Inspect the attached image", prompt)

    def test_visual_memory_survives_provider_routing_and_next_prompt(self):
        for provider in ("openrouter", "gemini", "openai"):
            with self.subTest(provider=provider), patch(
                f"nav.harness.llm_provider.call_{provider}",
                return_value=json.dumps({
                    "action": "forward", "reasoning": "The path is clear.",
                    "observation": "Forklift on the left; clear aisle ahead.",
                }),
            ) as call:
                captured = {"position": (168, 184), "world_position": (30.21, 54.69),
                            "theta": 180, "distance_to_target_m": 48.09}
                result = {}
                execute_decision("llm", {
                    "prompt": render_sample(), "images": [("ego", np.zeros((240, 320, 3), dtype=np.uint8))],
                    "model_id": "test-model", "llm_provider": provider,
                    "max_tokens": 20000, "allowed_actions": list(ACTION_SPACE_AGENTS),
                    "history_entry": captured,
                }, result)
                call.assert_called_once()
                self.assertTrue(result["finished"])
                self.assertFalse(result["error"])
                self.assertEqual(result["history_entry"], captured)
                history = deque(maxlen=5)
                for step in range(1, 7):
                    history.append({**result["history_entry"], "step": step,
                                    "action": result["action"], "observation": result["observation"]})
                next_prompt = render_sample(format_history_for_prompt(history))
                self.assertIn("Saw: Forklift on the left; clear aisle ahead.", next_prompt)
                self.assertNotIn("Step 1:", next_prompt)
                self.assertIn("Step 6:", next_prompt)

    def test_invalid_response_is_not_a_successful_stop(self):
        for raw in ("not json", '{"action":"fly"}', '{"action":"forward","observation":{}}',
                    '{"error":true,"reasoning":"upstream failure"}'):
            with self.subTest(raw=raw):
                self.assertTrue(parse_navigation_decision(raw, ACTION_SPACE_AGENTS)["error"])

    @patch("nav.harness.llm_provider.call_openrouter")
    def test_no_vision_does_not_store_hallucinated_visual_memory(self, call):
        call.return_value = '{"action":"stop","observation":"imaginary wall"}'
        decision = llm_generate_decision("navigate", [], "test", allowed_actions=ACTION_SPACE_AGENTS)
        self.assertEqual(decision["observation"], "")


class NavigationConfigTest(unittest.TestCase):
    def test_llm_defaults_and_other_baselines(self):
        for baseline, ego, minimap in (("llm", (320, 240), (431, 256)),
                                      ("astar", (512, 512), (862, 512)),
                                      ("bc", (512, 512), (862, 512))):
            with self.subTest(baseline=baseline):
                args = argparse.Namespace(ego_width=None, ego_height=None,
                                          minimap_width=None, minimap_height=None)
                resolve_navigation_sensors(args, baseline)
                self.assertEqual((args.ego_width, args.ego_height), ego)
                self.assertEqual((args.minimap_width, args.minimap_height), minimap)

    def test_explicit_resolution_and_single_axis_override(self):
        args = argparse.Namespace(ego_width=640, ego_height=480,
                                  minimap_width=None, minimap_height=512)
        resolve_navigation_sensors(args)
        self.assertEqual((args.ego_width, args.ego_height), (640, 480))
        self.assertEqual((args.minimap_width, args.minimap_height), (862, 512))

    def test_grid_and_cell_record_identical_configuration(self):
        with patch("sys.argv", ["grid", "--models", "test/model", "--seeds", "0"]):
            grid_args = grid_runner.parse_args()
        resolve_navigation_sensors(grid_args)
        cell = grid_runner.build_grid(["test/model"], ["scene1"], ["0"], [True], [5], ["point1"])[0]
        argv = ["cell", "--scene_id", "0", "--scene_name", "scene1", "--point_id", "point1",
                "--seed_id", "0", "--model_id", "test/model", "--file_name", "/tmp/fake-unity",
                "--init_world_x", str(cell.init_world_x), "--init_world_z", str(cell.init_world_z),
                "--init_curr_direction", str(cell.init_direction),
                "--target_x", str(cell.target_x), "--target_y", str(cell.target_y)]
        with patch("sys.argv", argv):
            cell_args = cell_runner.parse_args()
        resolve_navigation_sensors(cell_args)
        self.assertEqual(
            grid_runner.cell_run_config(grid_args, cell, "/tmp/fake-unity"),
            navigation_run_config(cell_args, load_prompt_template(DEFAULT_PROMPT_VISION)),
        )

    def test_old_runs_are_preserved_and_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old = root / "results.csv"
            old.write_text("old experiment\n")
            with self.assertRaisesRegex(ValueError, "fresh --output_root"):
                prepare_navigation_run(root, {"protocol": "new"})
            self.assertEqual(old.read_text(), "old experiment\n")
            self.assertFalse((root / "run_config.json").exists())

    @patch("nav.scripts.run_benchmark_grid.free_tcp_port", return_value=55555)
    @patch("nav.scripts.run_benchmark_grid.subprocess.run")
    def test_worker_command_preserves_effective_configuration(self, run, port):
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = ["grid", "--models", "test/model", "--seeds", "0",
                    "--output_root", tmpdir, "--motion_random_seed", "12",
                    "--light_random_seed", "15", "--light_fixed_exposure", "8"]
            with patch("sys.argv", argv):
                args = grid_runner.parse_args()
            resolve_navigation_sensors(args)
            cell = grid_runner.build_grid(["test/model"], ["scene1"], ["0"], [True], [5],
                                          ["point1"], output_root=tmpdir)[0]
            expected = grid_runner.cell_run_config(args, cell, "/tmp/fake-unity")

            def fake_subprocess(cmd, **kwargs):
                with patch("sys.argv", ["cell", *cmd[3:]]):
                    actual_args = cell_runner.parse_args()
                resolve_navigation_sensors(actual_args)
                actual = navigation_run_config(actual_args, load_prompt_template(actual_args.prompt_file))
                self.assertEqual(actual, expected)
                prepare_navigation_run(cell.frame_save_dir, actual)
                cell.results_csv.write_text("stop_reason\nmax_steps\n")
                return argparse.Namespace(returncode=0)

            run.side_effect = fake_subprocess
            result = grid_runner.run_cell({
                **vars(args), "cell": asdict(cell), "file_name": "/tmp/fake-unity",
                "use_xvfb": False,
            })
            self.assertTrue(result["ok"], result)
            run.assert_called_once()

    def test_matching_manifest_and_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = {"protocol": "kiro-aligned-v1", "settings": {"history_size": 5}}
            prepare_navigation_run(root, config)
            check_navigation_run_config(root, config)
            with self.assertRaises(ValueError):
                check_navigation_run_config(root, {**config, "settings": {"history_size": 10}})

    def test_api_errors_are_not_completed_grid_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cell = grid_runner.build_grid(["test/model"], ["scene1"], ["0"], [True], [5],
                                          ["point1"], output_root=tmpdir)[0]
            cell.frame_save_dir.mkdir(parents=True)
            for stop, expected in (("decision_error", False), ("max_steps", True),
                                   ("reached_vicinity", True), ("vision_unavailable", False)):
                with cell.results_csv.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["stop_reason"])
                    writer.writeheader()
                    writer.writerow({"stop_reason": stop})
                self.assertEqual(grid_runner.cell_completed(cell), expected)

    def test_manifest_does_not_leak_keys(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "do-not-record-this-secret"}):
            config = navigation_run_config(argparse.Namespace(), "prompt")
        self.assertNotIn("do-not-record-this-secret", json.dumps(config))


if __name__ == "__main__":
    unittest.main()
