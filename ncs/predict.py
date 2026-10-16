import os
import sys
import argparse
import tensorflow as tf

from dataset.data import Data
from model.ncs import NCS
from utils.config import MainConfig
from utils.inference_checkpoint import inference_checkpoint
from utils.IO import writePC2Frames
from global_vars import RESULTS_DIR

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(config, w=1.0, weights="best"):
    checkpoint = inference_checkpoint(config.name, weights)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Selected {weights} model weights not found: {checkpoint}"
        )
    model = NCS(config)
    model.build(input_shape=config.input_shape)
    model.load_weights(str(checkpoint))
    print(f"Loaded {weights} model: {checkpoint}")

    # Init data
    data = Data(config, mode="test")

    # Predict & store
    print("Predicting...")
    folder = os.path.join(RESULTS_DIR, config.name)
    # Filenames for results
    filenames = {
        "body": os.path.join(folder, "body.pc2"),
        "cloth": os.path.join(folder, config.garment.name + ".pc2"),
        "cloth_unskinned": os.path.join(folder, config.garment.name + "_unskinned.pc2"),
    }
    if w != 1.0:
        filenames["cloth"] = filenames["cloth"].replace(
            ".pc2", "_w{:.1f}".format(w) + ".pc2"
        )
        filenames["cloth_unskinned"] = filenames["cloth_unskinned"].replace(
            ".pc2", "_w{:.1f}".format(w) + ".pc2"
        )
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for filename in filenames.values():
        if os.path.isfile(filename):
            os.remove(filename)
    for i, batch in enumerate(data):
        sys.stdout.write("\r" + str(i + 1) + "/" + str(len(data)))
        sys.stdout.flush()
        body, cloth, unskinned = model.predict(batch, w=w)
        # Store results
        writePC2Frames(filenames["body"], body.numpy())
        writePC2Frames(filenames["cloth"], cloth.numpy())
        writePC2Frames(filenames["cloth_unskinned"], unskinned.numpy())
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, default="")
    parser.add_argument("--motion", type=float, default=1.0)
    parser.add_argument("--weights", choices=("best", "latest"), default="best",
                        help="Model weights to use for inference (default: best)")
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
    main(config, w=opts.motion, weights=opts.weights)
