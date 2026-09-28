"""Repeated fresh-player initialization must not stack observation decoders."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import mlagents_envs.rpc_utils as rpc
from mlagents_envs.communicator_objects.observation_pb2 import ObservationProto
from mlagents_envs.exception import UnityObservationException

from nav.harness.observations import patch_observation_decoding


class ObservationDecoderPatchTest(unittest.TestCase):
    def test_real_uncompressed_depth_and_rgb_are_unchanged(self):
        original = rpc._observation_to_np_array
        for shape in ([2, 2, 1], [2, 2, 3]):
            with self.subTest(shape=shape):
                observation = ObservationProto()
                observation.shape.extend(shape)
                observation.float_data.data.extend(
                    np.linspace(0, 1, int(np.prod(shape)), dtype=np.float32))
                expected = original(observation, shape)
                with patch.object(rpc, "_observation_to_np_array", original):
                    for _ in range(2000):
                        patch_observation_decoding()
                    actual = rpc._observation_to_np_array(observation, shape)
                    np.testing.assert_array_equal(actual, expected)

    def test_many_fresh_initializations_install_exactly_once(self):
        value = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        decoder = Mock(return_value=value)
        with patch.object(rpc, "_observation_to_np_array", decoder):
            patch_observation_decoding()
            installed = rpc._observation_to_np_array
            for _ in range(2000):
                patch_observation_decoding()
                self.assertIs(rpc._observation_to_np_array, installed)
            self.assertIs(installed("observation", (2, 2, 3)), value)
            decoder.assert_called_once_with("observation", (2, 2, 3))

    def test_concurrent_installation_preserves_one_decoder_call(self):
        decoder = Mock(return_value="decoded")
        with patch.object(rpc, "_observation_to_np_array", decoder):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: patch_observation_decoding(), range(1000)))
            self.assertEqual(rpc._observation_to_np_array("observation"), "decoded")
            decoder.assert_called_once_with("observation", None)

    def test_replaced_library_decoder_can_be_patched_again(self):
        first, second = Mock(return_value="first"), Mock(return_value="second")
        with patch.object(rpc, "_observation_to_np_array", first):
            patch_observation_decoding()
            self.assertEqual(rpc._observation_to_np_array(None), "first")
            rpc._observation_to_np_array = second
            patch_observation_decoding()
            self.assertEqual(rpc._observation_to_np_array(None), "second")
            first.assert_called_once()
            second.assert_called_once()

    def test_shape_tolerance_still_preserves_depth_and_rgb_fallbacks(self):
        for shape, channels in (([2, 2, 1], 1), ([3, 2, 2], 3)):
            with self.subTest(shape=shape):
                decoder = Mock(side_effect=UnityObservationException(
                    "Decompressed observation did not have the expected shape"))
                value = np.ones((2, 2, channels), dtype=np.float32)
                observation = SimpleNamespace(shape=shape, compressed_data=b"encoded",
                                              compressed_channel_mapping=[0, 1, 2])
                with patch.object(rpc, "_observation_to_np_array", decoder), \
                        patch.object(rpc, "process_pixels", return_value=value) as pixels:
                    for _ in range(1500):
                        patch_observation_decoding()
                    self.assertIs(rpc._observation_to_np_array(observation, shape), value)
                    decoder.assert_called_once_with(observation, shape)
                    pixels.assert_called_once_with(b"encoded", channels, [0, 1, 2])

    def test_other_decoding_errors_are_not_suppressed(self):
        for error in (UnityObservationException("invalid sensor"), ValueError("bad pixels")):
            with self.subTest(error=error):
                with patch.object(rpc, "_observation_to_np_array", Mock(side_effect=error)), \
                        patch.object(rpc, "process_pixels") as pixels:
                    patch_observation_decoding()
                    with self.assertRaises(type(error)) as caught:
                        rpc._observation_to_np_array(None)
                    self.assertIs(caught.exception, error)
                    pixels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
