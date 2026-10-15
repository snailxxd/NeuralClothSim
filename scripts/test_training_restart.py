"""CPU checks for OOM supervision and exact training checkpoint restoration."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ncs"))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.training_restart import OOM_EXIT_CODE, read_state, write_state, supervise


class RestartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ncs-restart-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = os.path.join(self.temp.name, "state.json")
        self.prefix = os.path.join(self.temp.name, "ckpt")
        Path(self.prefix + ".index").touch()

    def test_oom_resumes_committed_epoch(self):
        commands = []

        def worker(command):
            commands.append(command)
            if len(commands) == 1:
                write_state(self.path, 141, self.prefix)
                return subprocess.CompletedProcess(command, OOM_EXIT_CODE)
            self.assertIn("--recovery-state", command)
            self.assertEqual(read_state(command[-1])["completed_epoch"], 141)
            return subprocess.CompletedProcess(command, 0)

        with patch("utils.training_restart.subprocess.run", side_effect=worker):
            self.assertEqual(supervise("train.py", ["--config", "example.json"], self.path), 0)
        self.assertEqual(len(commands), 2)

    def test_non_oom_error_is_not_retried(self):
        with patch("utils.training_restart.subprocess.run", return_value=
                   subprocess.CompletedProcess([], 1)) as run:
            self.assertEqual(supervise("train.py", [], self.path), 1)
            self.assertEqual(run.call_count, 1)

    def test_oom_without_checkpoint_stops(self):
        with patch("utils.training_restart.subprocess.run", return_value=
                   subprocess.CompletedProcess([], OOM_EXIT_CODE)) as run:
            self.assertEqual(supervise("train.py", [], self.path), OOM_EXIT_CODE)
            self.assertEqual(run.call_count, 1)

    def test_no_progress_stops(self):
        write_state(self.path, 141, self.prefix)
        with patch("utils.training_restart.subprocess.run", return_value=
                   subprocess.CompletedProcess([], OOM_EXIT_CODE)) as run:
            self.assertEqual(supervise("train.py", [], self.path, resume=True), OOM_EXIT_CODE)
            self.assertEqual(run.call_count, 2)

    def test_restart_limit(self):
        write_state(self.path, 141, self.prefix)
        with patch("utils.training_restart.subprocess.run", return_value=
                   subprocess.CompletedProcess([], OOM_EXIT_CODE)) as run:
            self.assertEqual(supervise("train.py", [], self.path, max_restarts=0), OOM_EXIT_CODE)
            self.assertEqual(run.call_count, 1)

    def test_weights_adam_and_next_update_match(self):
        import numpy as np
        import tensorflow as tf
        from train import RecoveryCheckpoint

        def model():
            result = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(2,)),
                tf.keras.layers.Dense(1, kernel_initializer="ones"),
            ])
            result.compile(optimizer=tf.keras.optimizers.Adam(0.01, jit_compile=False))
            result.optimizer.build(result.trainable_variables)
            return result

        def update(m):
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(m(tf.constant([[1., 2.]])) ** 2)
            m.optimizer.apply_gradients(zip(tape.gradient(loss, m.trainable_variables),
                                            m.trainable_variables))

        original = model()
        update(original)
        recovery = RecoveryCheckpoint(self.temp.name, initial_epoch=140)
        recovery.set_model(original)
        recovery.on_train_begin()
        recovery.on_epoch_end(140)
        state = read_state(self.path)
        self.assertEqual(state["completed_epoch"], 141)
        restored = model()
        ckpt = tf.train.Checkpoint(model=restored, optimizer=restored.optimizer)
        ckpt.restore(state["checkpoint_prefix"]).assert_existing_objects_matched()
        for left, right in zip(original.weights + original.optimizer.variables(),
                               restored.weights + restored.optimizer.variables()):
            np.testing.assert_array_equal(left.numpy(), right.numpy())
        update(original)
        update(restored)
        for left, right in zip(original.weights, restored.weights):
            np.testing.assert_array_equal(left.numpy(), right.numpy())

    def test_ncs_saved_model_preserves_recovery_and_restores_adam(self):
        import numpy as np
        from train import BestModelCheckpoint, RecoveryCheckpoint, make_model
        from utils.config import MainConfig
        from dataset.data import Data

        config = MainConfig(str(ROOT / "ncs/configs/mannequin_tshirtloose_multi.json"))
        config.experiment.checkpoint = None
        config.experiment.batch_size = 1
        original = make_model(config)
        original.train_step(Data(config, mode="validation")[0])
        directory = os.path.join(self.temp.name, "recovery")
        recovery = RecoveryCheckpoint(directory)
        recovery.set_model(original)
        recovery.on_epoch_end(140)
        best = BestModelCheckpoint(self.temp.name)
        best.set_model(original)
        best.on_epoch_end(140, {best.monitor: 1.5})
        original.save(self.temp.name)
        config.recovery_state = read_state(os.path.join(directory, "state.json"))
        saved_learning_rate = float(original.optimizer.learning_rate.numpy())
        config.experiment.learning_rate = saved_learning_rate / 3
        restored = make_model(config)
        self.assertEqual(config.recovery_state["completed_epoch"], 141)
        self.assertAlmostEqual(
            float(restored.optimizer.learning_rate.numpy()),
            config.experiment.learning_rate,
        )
        self.assertNotEqual(
            float(restored.optimizer.learning_rate.numpy()), saved_learning_rate
        )
        for left, right in zip(original.weights + original.optimizer.variables(),
                               restored.weights + restored.optimizer.variables()):
            np.testing.assert_array_equal(left.numpy(), right.numpy())
        # A later SavedModel export must preserve the independent best weights.
        restored.load_weights(best.path)
        for left, right in zip(original.weights, restored.weights):
            np.testing.assert_array_equal(left.numpy(), right.numpy())


class BestCheckpointTests(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from train import BestModelCheckpoint

        self.temp = tempfile.TemporaryDirectory(prefix="ncs-best-test-")
        self.addCleanup(self.temp.cleanup)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="ones"),
        ])
        self.callback = BestModelCheckpoint(self.temp.name)
        self.callback.set_model(self.model)

    def test_best_survives_worse_epochs_nonfinite_values_and_restart(self):
        from train import BestModelCheckpoint

        self.callback.on_epoch_end(0, {self.callback.monitor: 2.0})
        self.model.weights[0].assign([[3.0]])
        self.callback.on_epoch_end(1, {self.callback.monitor: 1.0})
        resumed = BestModelCheckpoint(self.temp.name)
        resumed.set_model(self.model)
        self.assertEqual((resumed.best, resumed.best_epoch), (1.0, 2))
        self.model.weights[0].assign([[99.0]])
        for value in [1.0, 4.0, float("nan"), float("inf"), -float("inf")]:
            resumed.on_epoch_end(2, {resumed.monitor: value})
        self.model.load_weights(resumed.path)
        self.assertEqual(float(self.model.weights[0][0, 0]), 3.0)
        self.assertEqual(resumed.best_epoch, 2)

    def test_failed_save_preserves_previous_best(self):
        from train import BestModelCheckpoint

        self.callback.on_epoch_end(0, {self.callback.monitor: 2.0})
        self.model.weights[0].assign([[9.0]])

        def failed_save(path):
            Path(path).write_bytes(b"incomplete weights")
            raise RuntimeError("interrupted write")

        with patch.object(self.model, "save_weights", side_effect=failed_save):
            with self.assertRaisesRegex(RuntimeError, "interrupted write"):
                self.callback.on_epoch_end(1, {self.callback.monitor: 1.0})
        restored = BestModelCheckpoint(self.temp.name)
        self.assertEqual((restored.best, restored.best_epoch), (2.0, 1))
        self.model.load_weights(restored.path)
        self.assertEqual(float(self.model.weights[0][0, 0]), 1.0)
        self.assertFalse(Path(self.temp.name, ".best.pending.weights.h5").exists())

    def test_missing_validation_metric_is_reported(self):
        with self.assertRaisesRegex(ValueError, "val_m/Loss"):
            self.callback.on_epoch_end(0, {"loss": 1.0})


if __name__ == "__main__":
    unittest.main()
