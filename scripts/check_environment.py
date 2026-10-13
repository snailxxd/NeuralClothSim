"""Check dependencies, sample data, forward passes, and one training step."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ncs"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import ray
import scipy
import tensorflow as tf

from dataset.data import Data
from model.ncs import NCS
from utils.config import MainConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Python {sys.version.split()[0]}, TensorFlow {tf.__version__}, "
          f"NumPy {np.__version__}, SciPy {scipy.__version__}, Ray {ray.__version__}")
    print(f"GPUs: {gpus}")
    if args.require_gpu and not gpus:
        raise RuntimeError("No GPU visible to TensorFlow")
    for name in ("mixamo", "smpl"):
        tf.keras.backend.clear_session()
        config = MainConfig(str(ROOT / "ncs" / "configs" / f"{name}.json"))
        config.experiment.batch_size = 1
        model = NCS(config)
        model.build(config.input_shape)
        device = model.trainable_variables[0].device
        if args.require_gpu and "GPU:" not in device:
            raise RuntimeError(f"{name}: model variables are on {device}")
        model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate))
        batch = Data(config, mode="validation")[0]
        outputs = model(batch, training=False)
        for value in outputs:
            tf.debugging.assert_all_finite(value, f"{name}: invalid model output")
        # Collision detection uses SciPy, so the original model trains eagerly.
        metrics = model.train_step(batch)
        for key, value in metrics.items():
            tf.debugging.assert_all_finite(value, f"{name}: invalid {key}")
        poses, trans = batch
        predictions = model.predict((poses[0], trans[0]), w=1.0)
        for value in predictions:
            tf.debugging.assert_all_finite(value, f"{name}: invalid prediction")
        print(f"PASS {name}: sample data, model build, forward pass, "
              f"training step, prediction; device={device}; "
              f"loss={float(model.loss_metric.result()):.6f}")
        del model
    print("Environment check passed.")


if __name__ == "__main__":
    main()
