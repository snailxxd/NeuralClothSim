import os
import sys
import argparse
import logging

from global_vars import LOGS_DIR, CHECKPOINTS_DIR
from utils.training_state import read_state


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    parser.add_argument("--verbose", action="store_true", help="Show detailed TensorFlow logs, model summary and batch progress")
    parser.add_argument("--gpu-memory-limit-mb", type=int, default=None,
                        help="Optional per-GPU memory cap in MiB; default: allocate on demand without a fixed cap")
    args = parser.parse_args(argv)
    if args.gpu_memory_limit_mb is not None and args.gpu_memory_limit_mb <= 0:
        parser.error("--gpu-memory-limit-mb must be positive")
    return args


if __name__ == "__main__":
    opts = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id

VERBOSE = "--verbose" in sys.argv
# Keep native ERROR messages; INFO/WARNING startup diagnostics are optional.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0" if VERBOSE else "2")
import tensorflow as tf

from model.ncs import NCS
from dataset.data import Data
from utils.config import MainConfig
from utils.training_callbacks import BestModelCheckpoint, LossProgress, RecoveryCheckpoint


class TrainingLogFilter(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.ERROR:
            return True
        message = record.getMessage()
        # These are expected for this architecture and full-model checkpoints.
        return not (
            message.startswith("Model's `__init__()` arguments contain non-serializable objects.")
            or (message.startswith("Layer dyn_enc/gru ") and "generic GPU kernel" in message)
        )


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


def apply_recovery_state(config):
    """Restore the recorded epoch only for an explicitly requested same-name resume."""
    path = os.path.join(CHECKPOINTS_DIR, config.name, "recovery", "state.json")
    if config.experiment.checkpoint == config.name and os.path.isfile(path):
        state = read_state(path)
        config.recovery_state = state
        config.experiment.initial_epoch = state["completed_epoch"]


def main(config):
    apply_recovery_state(config)
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
    print(f"Experiment: {config.name}; garment: {config.garment.name}; "
          f"batch size: {config.experiment.batch_size}; learning rate: {config.learning_rate}")
    for label, dataset in (("Train", data), ("Validation", validation_data)):
        print(f"{label}: {dataset.num_sequences} sequences, {dataset.num_samples} samples, "
              f"{len(dataset)} batches/epoch")
    print(f"Epochs: {initial_epoch + 1}..{config.experiment.epochs}; "
          f"logs: {log_dir}; checkpoints: {checkpoint_dir}", flush=True)

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
            tf.keras.callbacks.CSVLogger(os.path.join(log_dir, "metrics.csv"), append=resume_in_place),
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
            LossProgress(count_mode="steps"),
        ],
    )


def configure_gpu_memory(gpus, memory_limit_mb=None):
    """Configure allocation before the first TensorFlow GPU operation."""
    for gpu in gpus:
        if memory_limit_mb is None:
            tf.config.experimental.set_memory_growth(gpu, True)
        else:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)],
            )
    if memory_limit_mb is None:
        print("TensorFlow GPU memory: growth enabled, no fixed allocation cap")
    else:
        print(f"TensorFlow GPU memory limit: {memory_limit_mb} MiB per GPU")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected")
        sys.exit(1)

    configure_gpu_memory(gpus, opts.gpu_memory_limit_mb)

    config = MainConfig(opts.config)
    try:
        main(config)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Resume manually from the last completed checkpoint.")
