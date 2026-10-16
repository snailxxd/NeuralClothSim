"""Checkpoint safety and compact output using the standard Keras progress bar."""
import math
import os

import h5py
import tensorflow as tf

from utils.training_state import write_state


class RecoveryCheckpoint(tf.keras.callbacks.Callback):
    """Save model + Adam + completed epoch without tracing a SavedModel."""

    def __init__(self, directory, initial_epoch=0, already_restored=False):
        super().__init__()
        self.directory = directory
        self.initial_epoch = initial_epoch
        self.already_restored = already_restored

    def set_model(self, model):
        super().set_model(model)
        self.checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, self.directory, max_to_keep=2
        )

    def save_epoch(self, completed_epoch):
        assert_finite_model(self.model)
        prefix = self.manager.save(checkpoint_number=completed_epoch)
        write_state(os.path.join(self.directory, "state.json"), completed_epoch,
                    os.path.abspath(prefix))

    def on_train_begin(self, logs=None):
        if not self.already_restored:
            # Establish a recovery snapshot before the first training epoch.
            self.save_epoch(self.initial_epoch)

    def on_epoch_end(self, epoch, logs=None):
        assert_finite_logs(logs)
        self.save_epoch(epoch + 1)


class BestModelCheckpoint(tf.keras.callbacks.Callback):
    """Atomically retain the lowest validation-loss weights across resumed runs."""

    monitor = "val_m/Loss"
    objective = "total_loss_v1"

    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.path = os.path.join(directory, "best.weights.h5")
        self.best = math.inf
        self.best_epoch = None
        if os.path.isfile(self.path):
            with h5py.File(self.path, "r") as saved:
                if saved.attrs.get("ncs_best_objective") == self.objective:
                    self.best = float(saved.attrs["ncs_best_val_loss"])
                    self.best_epoch = int(saved.attrs["ncs_best_epoch"])
                else:
                    print("Previous best weights used the old loss definition; "
                          "starting a new best-score comparison for total loss.", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or self.monitor not in logs:
            raise ValueError(f"Best model requires validation metric {self.monitor!r}")
        value = float(logs[self.monitor])
        if not math.isfinite(value) or value >= self.best:
            return
        assert_finite_model(self.model)
        os.makedirs(self.directory, exist_ok=True)
        pending = os.path.join(self.directory, ".best.pending.weights.h5")
        try:
            self.model.save_weights(pending)
            # Keep the score and epoch with their weights in a single commit.
            with h5py.File(pending, "r+") as saved:
                saved.attrs["ncs_best_val_loss"] = value
                saved.attrs["ncs_best_epoch"] = epoch + 1
                saved.attrs["ncs_best_objective"] = self.objective
            os.replace(pending, self.path)
        finally:
            if os.path.isfile(pending):
                os.remove(pending)
        self.best = value
        self.best_epoch = epoch + 1


def assert_finite_model(model):
    variables = list(model.weights)
    if getattr(model, "optimizer", None) is not None:
        variables += list(model.optimizer.variables())
    for variable in variables:
        if variable.dtype.is_floating:
            tf.debugging.assert_all_finite(variable, f"Refusing to save non-finite state: {variable.name}")


def assert_finite_logs(logs):
    for name, value in (logs or {}).items():
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"Non-finite epoch metric {name}={value}; checkpoint not saved")


class LossProgress(tf.keras.callbacks.ProgbarLogger):
    """Keep batch progress compact; print validation components once per epoch."""

    @staticmethod
    def compact(logs):
        return {key: value for key, value in (logs or {}).items()
                if key in ("m/Loss", "val_m/Loss")}

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, self.compact(logs))

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, self.compact(logs))
        if self.verbose:
            names = ("StaticLoss", "InertiaLoss", "ClothLoss", "BendingLoss",
                     "CollisionLoss", "Gravity", "PinningLoss")
            values = [f"{name}={float(logs['val_m/' + name]):.6g}"
                      for name in names if 'val_m/' + name in (logs or {})]
            for start in range(0, len(values), 3):
                print("  val: " + "  ".join(values[start:start + 3]), flush=True)
