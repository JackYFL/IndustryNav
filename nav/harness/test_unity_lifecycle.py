"""Teardown overrides must not modify observations or the connection timeout."""

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException

from nav.envs.unity_pointgoal import UnityPointGoalEnv
from nav.harness.unity_lifecycle import (
    CLOSE_TIMEOUT_ENV, close_unity_environment, configured_close_timeout,
)


class UnityLifecycleTest(unittest.TestCase):
    def test_configuration_defaults_and_invalid_values(self):
        self.assertIsNone(configured_close_timeout({}))
        self.assertIsNone(configured_close_timeout({CLOSE_TIMEOUT_ENV: " "}))
        self.assertEqual(configured_close_timeout({CLOSE_TIMEOUT_ENV: "5"}), 5.)
        for value in ("0", "-2", "nan", "inf", "five"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, CLOSE_TIMEOUT_ENV):
                configured_close_timeout({CLOSE_TIMEOUT_ENV: value})

    def test_default_calls_the_original_public_method(self):
        env = SimpleNamespace(close=Mock(), _close=Mock(), _loaded=True, _timeout_wait=120)
        close_unity_environment(env)
        env.close.assert_called_once_with()
        env._close.assert_not_called()
        self.assertEqual(env._timeout_wait, 120)

    def test_override_uses_close_timeout_and_reaps_only_owned_process(self):
        process = Mock(pid=1234)
        env = SimpleNamespace(close=Mock(), _close=Mock(), _loaded=True,
                              _timeout_wait=120, _process=process)
        close_unity_environment(env, timeout_seconds=5, logger=Mock())
        env._close.assert_called_once_with(timeout=5)
        env.close.assert_not_called()
        process.wait.assert_called_once_with(timeout=1.)
        self.assertEqual(env._timeout_wait, 120)

    def test_no_subprocess_and_public_adapter_are_supported(self):
        env = SimpleNamespace(close=Mock(), _close=Mock(), _loaded=True, _process=None)
        close_unity_environment(env, timeout_seconds=5, logger=Mock())
        env._close.assert_called_once_with(timeout=5)
        adapter = SimpleNamespace(close=Mock())
        close_unity_environment(adapter, timeout_seconds=5)
        adapter.close.assert_called_once_with()

    def test_unloaded_environment_retains_its_public_error(self):
        env = SimpleNamespace(_loaded=False, _close=Mock(),
                              close=Mock(side_effect=UnityEnvironmentException("unloaded")))
        with self.assertRaises(UnityEnvironmentException):
            close_unity_environment(env, timeout_seconds=5)
        env._close.assert_not_called()

    def test_adapter_releases_handle_even_when_public_close_raises(self):
        env = UnityPointGoalEnv.__new__(UnityPointGoalEnv)
        env.close_timeout_seconds = None
        env.logger = Mock()
        env.primed = SimpleNamespace(env=SimpleNamespace(close=Mock(side_effect=RuntimeError("close failed"))))
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            env.close()
        self.assertIsNone(env.primed)
        env.close()  # The wrapper's repeated close stays idempotent.

    def test_pointgoal_adapter_captures_override_before_launch_and_uses_it(self):
        with patch.dict("os.environ", {CLOSE_TIMEOUT_ENV: "5"}):
            env = UnityPointGoalEnv([dict(scene_name="scene1")], unity_path="unused",
                output_dir="unused", worker_id=0, base_port=45000,
                reward_fn=lambda *_args, **_kwargs: 0., logger=Mock())
        player = SimpleNamespace(close=Mock(), _close=Mock(), _loaded=True,
                                 _process=None, _timeout_wait=120)
        env.primed = SimpleNamespace(env=player)
        env.close()
        player._close.assert_called_once_with(timeout=5.)
        self.assertEqual(player._timeout_wait, 120)
        self.assertIsNone(env.primed)

    def test_bad_override_fails_before_any_player_launch(self):
        with patch.dict("os.environ", {CLOSE_TIMEOUT_ENV: "bad"}), patch(
                "nav.envs.unity_pointgoal.setup_and_prime") as setup:
            with self.assertRaisesRegex(ValueError, CLOSE_TIMEOUT_ENV):
                UnityPointGoalEnv([dict(scene_name="scene1")], unity_path="unused",
                    output_dir="unused", worker_id=0, base_port=45000,
                    reward_fn=lambda *_args, **_kwargs: 0., logger=Mock())
            setup.assert_not_called()

    def test_installed_mlagents_api_terminates_and_reaps_a_real_owned_child(self):
        # Exercise the actual installed _close API without a Unity launch.
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        env = UnityEnvironment.__new__(UnityEnvironment)
        env._loaded = True
        env._timeout_wait = 120
        env._communicator = Mock()
        env._process = process
        try:
            close_unity_environment(env, timeout_seconds=.03, logger=Mock())
            self.assertIsNotNone(process.returncode)
            self.assertIsNone(env._process)
            self.assertFalse(env._loaded)
            self.assertEqual(env._timeout_wait, 120)
            env._communicator.close.assert_called_once_with()
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
