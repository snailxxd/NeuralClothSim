import os
import sys
import argparse
import logging
import json
import math

from global_vars import LOGS_DIR, CHECKPOINTS_DIR
from utils.training_restart import OOM_EXIT_CODE, read_state, write_state, supervise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    parser.add_argument("--verbose", action="store_true", help="Show detailed TensorFlow logs, model summary and batch progress")
    parser.add_argument("--max-oom-restarts", type=int, default=100,
                        help="Maximum automatic GPU OOM restarts (default: 100; 0 disables retries)")
    parser.add_argument("--training-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--recovery-state", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_oom_restarts < 0:
        parser.error("--max-oom-restarts must be nonnegative")
    return args


# The supervisor must never initialize TensorFlow/CUDA: exiting the worker
# releases its complete GPU context before the next worker is started.
if __name__ == "__main__":
    opts = parse_args()
    if not opts.training_worker:
        with open(opts.config) as stream:
            settings = json.load(stream)
        name = os.path.basename(os.path.splitext(opts.config)[0])
        state_path = os.path.join(CHECKPOINTS_DIR, name, "recovery", "state.json")
        resume = (settings["experiment"].get("checkpoint") == name
                  and os.path.isfile(state_path))
        sys.exit(supervise(os.path.abspath(__file__), sys.argv[1:], state_path,
                           resume=resume, max_restarts=opts.max_oom_restarts))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id

VERBOSE = "--verbose" in sys.argv
# Keep native ERROR messages; INFO/WARNING startup diagnostics are optional.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0" if VERBOSE else "2")
import tensorflow as tf
import h5py

from model.ncs import NCS
from dataset.data import Data
from utils.config import MainConfig


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
        prefix = self.manager.save(checkpoint_number=completed_epoch)
        write_state(os.path.join(self.directory, "state.json"), completed_epoch,
                    os.path.abspath(prefix))

    def on_train_begin(self, logs=None):
        if not self.already_restored:
            # Also cover OOM in the first resumed epoch of a legacy checkpoint.
            self.save_epoch(self.initial_epoch)

    def on_epoch_end(self, epoch, logs=None):
        self.save_epoch(epoch + 1)


class BestModelCheckpoint(tf.keras.callbacks.Callback):
    """Atomically retain the lowest validation-loss weights across restarts."""

    monitor = "val_m/Loss"

    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.path = os.path.join(directory, "best.weights.h5")
        self.best = math.inf
        self.best_epoch = None
        if os.path.isfile(self.path):
            with h5py.File(self.path, "r") as saved:
                self.best = float(saved.attrs["ncs_best_val_loss"])
                self.best_epoch = int(saved.attrs["ncs_best_epoch"])

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or self.monitor not in logs:
            raise ValueError(f"Best model requires validation metric {self.monitor!r}")
        value = float(logs[self.monitor])
        if not math.isfinite(value) or value >= self.best:
            return
        os.makedirs(self.directory, exist_ok=True)
        pending = os.path.join(self.directory, ".best.pending.weights.h5")
        try:
            self.model.save_weights(pending)
            # Keep the score and epoch with their weights in a single commit.
            with h5py.File(pending, "r+") as saved:
                saved.attrs["ncs_best_val_loss"] = value
                saved.attrs["ncs_best_epoch"] = epoch + 1
            os.replace(pending, self.path)
        finally:
            if os.path.isfile(pending):
                os.remove(pending)
        self.best = value
        self.best_epoch = epoch + 1
        print(f"Best model saved: epoch {self.best_epoch}, "
              f"{self.monitor}={self.best:.6g} -> {self.path}", flush=True)


class TrainingLogFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        # These are expected for this architecture and full-model checkpoints.
        return not (
            message.startswith("Model's `__init__()` arguments contain non-serializable objects.")
            or (message.startswith("Layer dyn_enc/gru ") and "generic GPU kernel" in message)
        )


if not VERBOSE:
    tf.get_logger().addFilter(TrainingLogFilter())


def make_model(config):
    model = NCS(config)
    print("Building model...")
    model.build(input_shape=config.input_shape)
    if VERBOSE:
        model.summary()
    print("Compiling model...")
    # The model trains eagerly for SciPy collision detection. Avoid compiling
    # an XLA optimizer update for each differently shaped parameter tensor.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate, jit_compile=False
    )
    model.compile(optimizer=optimizer)
    # Include Adam slots even in the initial epoch-0 recovery snapshot.
    optimizer.build(model.trainable_variables)
    recovery_state = getattr(config, "recovery_state", None)
    if recovery_state is not None:
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        status = checkpoint.restore(recovery_state["checkpoint_prefix"])
        status.assert_existing_objects_matched()
        status.expect_partial()
        # Recovery owns learned state (weights, Adam moments and iteration),
        # while the current config owns runtime hyperparameters. TensorFlow
        # checkpoints also contain Adam's learning-rate variable, so restore it
        # first and then explicitly apply the possibly updated config value.
        optimizer.learning_rate.assign(config.learning_rate)
    elif config.experiment.checkpoint is not None:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, config.experiment.checkpoint)
        model.load_weights(checkpoint_path)
    return model


def main(config):
    initial_epoch = getattr(config.experiment, "initial_epoch", 0)
    if not isinstance(initial_epoch, int) or not 0 <= initial_epoch <= config.experiment.epochs:
        raise ValueError("initial_epoch must be an integer between 0 and epochs")
    if initial_epoch and config.experiment.checkpoint is None:
        raise ValueError("initial_epoch requires a checkpoint to resume from")
    # An explicit same-name resume continues writing to the existing outputs.
    # Other runs must use a new config name to protect their artifacts.
    log_dir = os.path.join(LOGS_DIR, config.name)
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, config.name)
    existing_dirs = [path for path in (log_dir, checkpoint_dir) if os.path.isdir(path)]
    resume_in_place = (
        (initial_epoch > 0 or getattr(config, "recovery_state", None) is not None)
        and config.experiment.checkpoint == config.name
        and (os.path.isfile(os.path.join(checkpoint_dir, "saved_model.pb"))
             or getattr(config, "recovery_state", None) is not None)
    )
    if existing_dirs and not resume_in_place:
        print(f"Experiment '{config.name}' cannot start because these output directories already exist:")
        for path in existing_dirs:
            print(f"  {path}")
        print("Rename/archive the listed directories, or use a different config filename.")
        raise FileExistsError("Experiment output directories already exist")

    print("Initializing model...")
    if len(gpus) > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = make_model(config)
    else:
        model = make_model(config)

    print("Reading data...")
    data = Data(config, mode="train")
    validation_data = Data(config, mode="validation")

    print("Training...")
    if config.experiment.checkpoint is not None:
        print(f"Loaded checkpoint '{config.experiment.checkpoint}'; starting at epoch {initial_epoch + 1}.")
    model.fit(
        data,
        verbose=1 if VERBOSE else 2,
        validation_data=validation_data,
        epochs=config.experiment.epochs,
        initial_epoch=initial_epoch,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                write_graph=False,
                write_steps_per_second=False,
                update_freq="epoch",
            ),
            # Commit a separate recoverable checkpoint before SavedModel
            # export, so even an OOM during export cannot destroy recovery.
            RecoveryCheckpoint(os.path.join(checkpoint_dir, "recovery"),
                               initial_epoch=initial_epoch,
                               already_restored=getattr(config, "recovery_state", None) is not None),
            BestModelCheckpoint(checkpoint_dir),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir, save_freq="epoch"
            ),
        ],
    )


if __name__ == "__main__":
    # Set the TensorFlow allocation limit before initializing any GPU.
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected")
        sys.exit(1)

    memory_limit_mb = 6144  # 6 GiB, approximately 75% of an 8 GiB GPU.
    for gpu in gpus:
        tf.config.set_logical_device_configuration(
            gpu,
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)],
        )
    print(f"TensorFlow GPU memory limit: {memory_limit_mb} MB per GPU")

    config = MainConfig(opts.config)
    if opts.recovery_state:
        state = read_state(opts.recovery_state)
        config.recovery_state = state
        config.experiment.checkpoint = config.name
        config.experiment.initial_epoch = state["completed_epoch"]
    try:
        main(config)
    except tf.errors.ResourceExhaustedError as error:
        message = str(error)
        if "GPU" not in message or not any(
            marker in message.lower() for marker in ("oom", "out of memory", "failed to allocate")
        ):
            raise
        print(f"GPU out of memory: {error}", file=sys.stderr, flush=True)
        sys.exit(OOM_EXIT_CODE)
