"""Export a training-only archive for one configuration, without changing Git branches."""
import argparse
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def training_files(config_path, include_data=False):
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    files = {config_path, ROOT / "LICENSE.md", ROOT / "requirements.txt",
             ROOT / "requirements-gpu.txt", ROOT / "ncs/train.py", ROOT / "ncs/global_vars.py"}
    for folder in ("model", "loss", "dataset", "utils"):
        files.update((ROOT / "ncs" / folder).glob("*.py"))
    files.discard(ROOT / "ncs/model/streaming.py")
    files.discard(ROOT / "ncs/utils/inference_checkpoint.py")
    body = ROOT / "body_models" / config["body"]["model"]
    files.add(body / "body.npz")
    files.add(ROOT / "body_models" / (config["body"]["skeleton"] + "_skeleton.json"))
    garment = body / (config["garment"]["name"] + ".obj")
    files.add(garment)
    pin = garment.with_name(garment.stem + "_pin.npy")
    if pin.is_file():
        files.add(pin)
    dataset = config["data"]["dataset"]
    lists = ROOT / "ncs/dataset/txt" / dataset
    metadata = lists / "preprocessing.json"
    if metadata.is_file():
        files.add(metadata)
    for mode in ("train", "validation", "test"):
        listing = lists / config["data"][mode]
        files.add(listing)
        if include_data:
            for name in listing.read_text(encoding="utf-8").splitlines():
                if not name.strip():
                    raise ValueError(f"Empty sequence name in {listing}")
                sequence = (ROOT / "data" / dataset / name).resolve()
                sequence.relative_to((ROOT / "data" / dataset).resolve())
                files.add(sequence)
    for path in files:
        path.resolve().relative_to(ROOT)
        if not path.is_file():
            raise FileNotFoundError(f"Required training file missing: {path}")
    return sorted(files), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "ncs/configs/smpl_tshirt.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-data", action="store_true",
                        help="Include exactly the sequences referenced by train/validation/test lists")
    args = parser.parse_args()
    files, config = training_files(args.config.resolve(), args.include_data)
    config_relative = args.config.resolve().relative_to(ROOT / "ncs")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    readme = f"""# NeuralClothSim training package

Python 3.11 / Linux with an NVIDIA GPU. Extract into an empty project directory.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gpu.txt
cd ncs
python train.py --config {config_relative.as_posix()} --gpu_id 0
```

Configuration: {args.config.name}
Motion data included: {args.include_data}
Without included data, separately copy data/{config['data']['dataset']}/ here.
This archive excludes visualization, prediction entry points, development scripts,
environments, logs and checkpoints. A configured resume checkpoint must be copied
separately, or set experiment.checkpoint to null for a new run.

GPU memory grows on demand with no fixed cap (suitable for a 96 GB H20).
Optional per-GPU cap: --gpu-memory-limit-mb 81920 (80 GiB).
Training runs directly in one process; OOM raises an error and exits.
Resume manually using the saved checkpoint after adjusting memory usage.
Training writes logs/ and checkpoints/ under this directory.
Add --verbose to show the existing batch progress bar.
Epoch metrics: logs/{args.config.stem}/metrics.csv (zero-based epoch).
m/Loss includes inertia; StaticLoss and weighted components are logged separately.
Non-finite loss/gradients abort before updating weights; checkpoint state is checked
before saving. Recovery paths are portable.
Batch progress shows total loss; validation components print once per epoch.
Full metrics stay in CSV/TensorBoard.
Best weights: checkpoints/{args.config.stem}/best.weights.h5
Resume: set experiment.checkpoint to "{args.config.stem}" and keep recovery/.
Run a one-epoch smoke experiment using a separate configuration filename first.

Code is from NeuralClothSim; see LICENSE.md. Body and motion assets retain their
original usage terms.
"""
    try:
        with tarfile.open(args.output, "x:gz", compresslevel=1) as archive:
            for path in files:
                archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
            contents = readme.encode("utf-8")
            info = tarfile.TarInfo("README.md")
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(f"Created {args.output}: {len(files) + 1} files, {args.output.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
