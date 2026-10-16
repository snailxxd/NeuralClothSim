"""Numerical checks; set NCS_TEST_GPU=1 for a real dress-training memory test."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ncs"))

import numpy as np
import tensorflow as tf

from model.layers import FullyConnected
from model.ncs import NCS

GPU_TEST = os.environ.get("NCS_TEST_GPU") == "1"
if GPU_TEST:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


class NumericalTests(unittest.TestCase):
    def test_fully_connected_matches_broadcast_outputs_and_gradients(self):
        for shape in [(2, 3, 5), (2, 3, 4, 5)]:
            with self.subTest(shape=shape):
                x = tf.Variable(tf.random.stateless_normal(shape, seed=[1, 2]))
                layer = FullyConnected(7, act=tf.nn.tanh)
                layer(x)
                with tf.GradientTape() as tape:
                    expected = tf.nn.tanh((x[..., None, :] @ layer.kernel)[..., 0, :]
                                          + layer.bias)
                    loss = tf.reduce_sum(expected ** 2)
                expected_gradients = tape.gradient(loss, [x, *layer.trainable_variables])
                with tf.GradientTape() as tape:
                    actual = layer(x)
                    loss = tf.reduce_sum(actual ** 2)
                gradients = tape.gradient(loss, [x, *layer.trainable_variables])
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
                    np.testing.assert_allclose(actual_gradient, expected_gradient,
                                               rtol=1e-5, atol=1e-6)

    def test_dynamic_graph_matches_eager_outputs_and_gradients(self):
        class Encoder(tf.keras.Model):
            encode_dynamic = NCS.encode_dynamic

            def __init__(self):
                super().__init__()
                self.dynamic_encoder = [
                    FullyConnected(6, act=tf.nn.relu, use_bias=False),
                    tf.keras.layers.GRU(8, use_bias=False, return_sequences=True),
                ]

        model = Encoder()
        for batch_size in (3, 1):
            for training in (True, False):
                with self.subTest(batch_size=batch_size, training=training):
                    x = tf.Variable(tf.random.stateless_normal([batch_size, 7, 5], [3, 4]))
                    with tf.GradientTape() as tape:
                        expected = x
                        for layer in model.dynamic_encoder:
                            expected = layer(expected, training=training)
                        loss = tf.reduce_sum(expected ** 2)
                    expected_gradients = tape.gradient(loss, [x, *model.trainable_variables])
                    with tf.GradientTape() as tape:
                        actual = model.encode_dynamic(x, training=training)
                        loss = tf.reduce_sum(actual ** 2)
                    gradients = tape.gradient(loss, [x, *model.trainable_variables])
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                    for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
                        np.testing.assert_allclose(actual_gradient, expected_gradient,
                                                   rtol=1e-5, atol=1e-6)


class TrainingMemoryTests(unittest.TestCase):
    @unittest.skipUnless(GPU_TEST, "enable with NCS_TEST_GPU=1")
    def test_dress_fit_and_checkpoints_keep_gpu_memory_bounded(self):
        self.assertTrue(tf.config.list_physical_devices("GPU"), "GPU required for this test")
        from dataset.data import Data
        from train import BestModelCheckpoint, RecoveryCheckpoint, make_model
        from utils.config import MainConfig

        config = MainConfig(str(ROOT / "ncs/configs/mannequin_dress_multi.json"))
        config.experiment.checkpoint = None
        config.experiment.batch_size = 4
        model = make_model(config)
        batch = Data(config, mode="validation")[0]

        class Batches(tf.keras.utils.Sequence):
            def __len__(self):
                return 3

            def __getitem__(self, index):
                return batch

        samples = []

        class Memory(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                current = tf.config.experimental.get_memory_info("GPU:0")["current"]
                samples.append(current)
                print(f"GPU epoch {epoch + 1}: {current / 2**20:.2f} MiB", flush=True)

        with tempfile.TemporaryDirectory(prefix="ncs-memory-test-") as directory:
            model.fit(
                Batches(), validation_data=Batches(), epochs=24, verbose=0,
                callbacks=[
                    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(directory, "logs"),
                                                   write_graph=False, update_freq="epoch"),
                    RecoveryCheckpoint(os.path.join(directory, "recovery")),
                    BestModelCheckpoint(directory),
                    tf.keras.callbacks.ModelCheckpoint(filepath=directory, save_freq="epoch"),
                    Memory(),
                ],
            )
        # Allow allocator rounding after warmup, but catch the former 512 KiB
        # leak on every forward pass (six passes per epoch in this test).
        steady = samples[3:]
        self.assertLessEqual(max(steady) - min(steady), 8 * 2**20, samples)


if __name__ == "__main__":
    unittest.main()
