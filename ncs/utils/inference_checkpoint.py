from pathlib import Path

from global_vars import CHECKPOINTS_DIR


def inference_checkpoint(experiment_name, weights):
    """Return the selected inference weights for an experiment."""
    experiment = Path(CHECKPOINTS_DIR) / experiment_name
    if weights == "best":
        return experiment / "best.weights.h5"
    if weights == "latest":
        return experiment
    raise ValueError(f"Unknown inference weights selection: {weights!r}")
