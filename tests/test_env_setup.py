import unittest
from types import SimpleNamespace

from nav.harness.env_setup import _set_world_spawn_parameters


class _FakeEnvironmentParameters:
    def __init__(self):
        self.values = {}

    def set_float_parameter(self, name, value):
        self.values[name] = value


class EnvSetupTest(unittest.TestCase):
    def test_world_spawn_can_be_seeded_before_initial_reset(self):
        params = _FakeEnvironmentParameters()
        args = SimpleNamespace(
            init_world_x=14.2,
            init_world_z=46.35,
            init_curr_direction=90.0,
        )

        self.assertTrue(_set_world_spawn_parameters(params, args))
        self.assertEqual(params.values, {
            "spawn_x": 14.2,
            "spawn_y": 0.5,
            "spawn_z": 46.35,
            "spawn_rot": 90.0,
        })

    def test_pixel_spawn_does_not_seed_incomplete_world_coordinates(self):
        params = _FakeEnvironmentParameters()
        args = SimpleNamespace(init_world_x=None, init_world_z=None)

        self.assertFalse(_set_world_spawn_parameters(params, args))
        self.assertEqual(params.values, {})


if __name__ == "__main__":
    unittest.main()
