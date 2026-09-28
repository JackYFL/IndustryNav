from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nav.harness.env_setup as env_setup

from nav.harness.env_setup import (
    EnvSetupError,
    _unity_threading_args,
    unity_build_scene_count,
    validate_unity_scene_id,
)


class UnityBuildSceneValidationTest(unittest.TestCase):
    def test_macos_build_counts_serialized_levels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            app = Path(temporary) / "scene_all.app"
            data = app / "Contents" / "Resources" / "Data"
            data.mkdir(parents=True)
            for index in range(3):
                (data / f"level{index}").touch()
            (data / "globalgamemanagers").touch()
            self.assertEqual(unity_build_scene_count(app), 3)
            self.assertEqual(validate_unity_scene_id(app, 2), 3)

    def test_out_of_range_scene_is_rejected_instead_of_clamped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "scene_all.x86_64"
            executable.touch()
            data = executable.parent / "scene_all_Data"
            data.mkdir()
            for index in range(12):
                (data / f"level{index}").touch()
            with self.assertRaisesRegex(
                EnvSetupError, "contains 12 scenes.*scene_id=20"
            ):
                validate_unity_scene_id(executable, 20)

    def test_unknown_editor_or_mock_path_remains_supported(self) -> None:
        self.assertIsNone(validate_unity_scene_id("mock-unity", 99))


class UnityThreadingArgumentsTest(unittest.TestCase):
    def test_defaults_do_not_change_player_flags(self):
        self.assertEqual(_unity_threading_args({}), [])
        self.assertEqual(_unity_threading_args({"INDUSTRYNAV_UNITY_JOB_WORKERS": "  "}), [])

    def test_worker_cap_and_direct_render_are_independent(self):
        workers = {"INDUSTRYNAV_UNITY_JOB_WORKERS": " 2 "}
        self.assertEqual(_unity_threading_args(workers), ["-job-worker-count", "2"])
        direct = {"INDUSTRYNAV_UNITY_SINGLE_THREADED_RENDER": "1"}
        self.assertEqual(_unity_threading_args(direct), ["-force-gfx-direct"])
        self.assertEqual(_unity_threading_args(dict(workers, **direct)),
                         ["-job-worker-count", "2", "-force-gfx-direct"])

    def test_invalid_worker_limits_fail_before_player_launch(self):
        for value in ("0", "-1", "2.5", "abc", "1 --nographics"):
            with self.subTest(value=value), self.assertRaisesRegex(EnvSetupError, "positive integer"):
                _unity_threading_args({"INDUSTRYNAV_UNITY_JOB_WORKERS": value})

    def test_render_mode_requires_explicit_zero_or_one(self):
        for value in ("", "false", "true", "2"):
            with self.subTest(value=value), self.assertRaisesRegex(EnvSetupError, "0 or 1"):
                _unity_threading_args({"INDUSTRYNAV_UNITY_SINGLE_THREADED_RENDER": value})

    def test_launcher_passes_diagnostic_flags_to_unity(self):
        speed = SimpleNamespace(speed_mps=1.)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(env_setup.os.environ, {
                    "INDUSTRYNAV_UNITY_JOB_WORKERS": "2",
                    "INDUSTRYNAV_UNITY_SINGLE_THREADED_RENDER": "1",
                    "INDUSTRYNAV_UNITY_BATCHMODE": "0",
                    "INDUSTRYNAV_UNITY_DEVICE_INDEX": "0",
                }), \
                patch.object(env_setup, "UnityEnvironment") as unity, \
                patch.object(env_setup, "configure_unity_motion_speed", return_value=
                    SimpleNamespace(human=speed, vehicle=speed, robot=speed)), \
                patch.object(env_setup, "configure_unity_lighting", return_value=SimpleNamespace(mode="disabled")):
            args = SimpleNamespace(file_name="mock-unity", scene_id=0, frame_save_dir=temporary,
                screen_width=1724, screen_height=1024, worker_id=0, base_port=5500)
            env_setup._launch_env(args, Mock())
            flags = unity.call_args.kwargs["additional_args"]
            self.assertEqual(flags[flags.index("-job-worker-count") + 1], "2")
            self.assertIn("-force-gfx-direct", flags)
            self.assertFalse(unity.call_args.kwargs["no_graphics"])
            unity.return_value.reset.assert_called_once()


class ResumeTargetPixelTest(unittest.TestCase):
    def test_resume_uses_acknowledged_pixel_without_redetecting_target(self):
        args = SimpleNamespace(target_x=570, target_y=150, minimap_width=431,
            minimap_height=256, _resume_target_unity_pixel=[285, 75])
        parameters = Mock()
        channel = SimpleNamespace(last_target_world=(10., 20.), last_target_pixel=(285, 75))
        with patch.object(env_setup, "visual_to_unity_coords") as convert:
            target = env_setup._prime_target(Mock(), parameters, channel,
                (10, 400, 20, 230), (285, 75), args, Mock())
        convert.assert_not_called()
        self.assertEqual(target, (10., 20.))
        parameters.set_float_parameter.assert_any_call("target_px", 285.)
        parameters.set_float_parameter.assert_any_call("target_py", 75.)

    def test_fresh_runs_keep_existing_conversion(self):
        args = SimpleNamespace(target_x=570, target_y=150, minimap_width=431,
            minimap_height=256)
        channel = SimpleNamespace(last_target_world=(10., 20.), last_target_pixel=(285, 75))
        with patch.object(env_setup, "visual_to_unity_coords", return_value=(285, 75)) as convert:
            env_setup._prime_target(Mock(), Mock(), channel, (0, 431, 0, 256),
                (285, 75), args, Mock())
        convert.assert_called_once()

    def test_invalid_saved_pixels_fail_before_unity_reset(self):
        for pixel in ([1], [1, 2, 3], [float("nan"), 2], None):
            if pixel is None:
                continue
            env = Mock()
            args = SimpleNamespace(_resume_target_unity_pixel=pixel)
            with self.subTest(pixel=pixel), self.assertRaises(EnvSetupError):
                env_setup._prime_target(env, Mock(), Mock(), None, None, args, Mock())
            env.reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
