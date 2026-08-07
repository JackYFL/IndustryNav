import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from nav.utils import (
    decode_depth_observation_meters,
    save_frame_from_obs,
    srgb_to_linear,
)


class DepthUtilsTest(unittest.TestCase):
    def test_srgb_to_linear_reference_values(self):
        encoded = np.array([0.0, 0.04045, 1.0], dtype=np.float32)
        expected = np.array([0.0, 0.0031308, 1.0], dtype=np.float32)
        np.testing.assert_allclose(srgb_to_linear(encoded), expected, rtol=1e-5)

    def test_decode_depth_uses_shader_max_distance(self):
        obs = np.array([[[0.0, 1.0]]], dtype=np.float32)
        np.testing.assert_allclose(
            decode_depth_observation_meters(obs),
            np.array([[0.0, 20.0]], dtype=np.float32),
        )

    def test_depth_save_preserves_fixed_png_scale_and_writes_meter_npy(self):
        obs = np.array([[[0.2, 0.4]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_frame_from_obs(obs, tmp_dir, "7.png")

            png = np.asarray(Image.open(Path(tmp_dir) / "7.png"))
            np.testing.assert_array_equal(png, np.array([[51, 102]], dtype=np.uint8))

            depth_m = np.load(Path(tmp_dir) / "7.npy")
            np.testing.assert_allclose(
                depth_m,
                srgb_to_linear(obs[0]) * 20.0,
                rtol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
