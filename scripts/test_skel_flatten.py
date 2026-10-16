"""Regression checks for skeleton flattening with generalized graph shapes."""
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ncs"))

import numpy as np
import tensorflow as tf

from model.layers import FullyConnected, SkelFlatten


class SkelFlattenTests(unittest.TestCase):
    def test_unknown_batch_and_time_preserve_values_and_feature_width(self):
        flatten = SkelFlatten()
        dense = FullyConnected(7)

        @tf.function(input_signature=[tf.TensorSpec([None, None, 16, 12], tf.float32)])
        def encode(x):
            flat = flatten(x)
            return flat, dense(flat)

        for batch, time in ((2, 5), (1, 32), (3, 0)):
            values = np.arange(batch * time * 16 * 12, dtype=np.float32)
            x = values.reshape(batch, time, 16, 12)
            flat, encoded = encode(x)
            np.testing.assert_array_equal(flat.numpy(), x.reshape(batch, time, 192))
            self.assertEqual(encoded.shape, (batch, time, 7))
        self.assertEqual(dense.kernel.shape, (192, 7))

    def test_retracing_generalizes_batch_and_time(self):
        flatten = SkelFlatten()

        @tf.function(reduce_retracing=True)
        def encode(x):
            return flatten(x)

        for shape in ((2, 5, 16, 12), (1, 5, 16, 12), (1, 32, 16, 12)):
            x = tf.reshape(tf.range(np.prod(shape), dtype=tf.float32), shape)
            np.testing.assert_array_equal(encode(x).numpy(), x.numpy().reshape(*shape[:2], 192))


if __name__ == "__main__":
    unittest.main()
