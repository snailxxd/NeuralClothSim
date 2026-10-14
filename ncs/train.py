import os
import sys
import argparse
import logging

VERBOSE = "--verbose" in sys.argv
# Keep native ERROR messages; INFO/WARNING startup diagnostics are optional.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0" if VERBOSE else "2")
import tensorflow as tf

from model.ncs import NCS
from dataset.data import Data
from utils.config import MainConfig
from global_vars import LOGS_DIR, CHECKPOINTS_DIR


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
    if config.experiment.checkpoint is not None:
        # Create Adam slots before restoring to recover its complete state.
        optimizer.build(model.trainable_variables)
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
        initial_epoch > 0
        and config.experiment.checkpoint == config.name
        and os.path.isfile(os.path.join(checkpoint_dir, "saved_model.pb"))
    )
    if existing_dirs and not resume_in_place:
        print(f"Experiment '{config.name}' cannot start because these output directories already exist:")
        for path in existing_dirs:
            print(f"  {path}")
        print("Rename/archive the listed directories, or use a different config filename.")
        return

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
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir, save_freq="epoch"
            ),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    parser.add_argument("--verbose", action="store_true", help="Show detailed TensorFlow logs, model summary and batch progress")
    opts = parser.parse_args()

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id

    # Limit VRAM usage
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not gpus:
        print("No GPU detected")
        sys.exit()

    config = MainConfig(opts.config)
    main(config)
