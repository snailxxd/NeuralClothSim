"""Read and atomically commit checkpoint metadata for manual training resumes."""
import json
import os
import re
from pathlib import Path


def read_state(path):
    path = Path(path)
    with open(path) as stream:
        state = json.load(stream)
    epoch = state["completed_epoch"]
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Invalid completed_epoch in recovery state")
    prefix = Path(state["checkpoint_prefix"])
    if not prefix.is_absolute():
        prefix = path.parent / prefix
    elif not Path(str(prefix) + ".index").is_file():
        # Older states stored absolute paths. Support a moved recovery folder.
        prefix = path.parent / prefix.name
    prefix = str(prefix.resolve())
    if not os.path.isfile(prefix + ".index"):
        raise ValueError("Recovery checkpoint index is missing")
    shards = list(Path(prefix).parent.glob(Path(prefix).name + ".data-*-of-*"))
    parts = [re.search(r"\.data-(\d+)-of-(\d+)$", shard.name) for shard in shards]
    if (not parts or any(part is None for part in parts)
            or len({int(part[2]) for part in parts}) != 1
            or {int(part[1]) for part in parts} != set(range(int(parts[0][2])))):
        raise ValueError("Recovery checkpoint data shards are missing or incomplete")
    state["checkpoint_prefix"] = prefix
    return state


def write_state(path, epoch, prefix):
    # Publish only after the checkpoint write has returned successfully.
    path = os.fspath(path)
    temporary = path + ".tmp"
    relative_prefix = os.path.relpath(prefix, os.path.dirname(os.path.abspath(path)))
    with open(temporary, "w") as stream:
        json.dump({"completed_epoch": epoch, "checkpoint_prefix": relative_prefix}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

