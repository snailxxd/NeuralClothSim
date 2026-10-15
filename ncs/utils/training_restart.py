"""Process supervision and committed recovery state (no TensorFlow import)."""
import json
import os
import subprocess
import sys

OOM_EXIT_CODE = 86


def read_state(path):
    with open(path) as stream:
        state = json.load(stream)
    epoch = state["completed_epoch"]
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Invalid completed_epoch in recovery state")
    prefix = state["checkpoint_prefix"]
    if not os.path.isfile(prefix + ".index"):
        raise ValueError("Recovery checkpoint index is missing")
    return state


def write_state(path, epoch, prefix):
    # Publish only after the checkpoint write has returned successfully.
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"completed_epoch": epoch, "checkpoint_prefix": prefix}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def supervise(script, arguments, state_path, resume=False, max_restarts=100):
    command = [sys.executable, "-u", script, *arguments, "--training-worker"]
    committed = read_state(state_path)["completed_epoch"] if resume else None
    restarts = 0
    stalled = 0
    while True:
        worker_command = command + (["--recovery-state", state_path] if resume else [])
        try:
            result = subprocess.run(worker_command)
        except KeyboardInterrupt:
            # The child shares the terminal process group and receives Ctrl-C.
            return 130
        if result.returncode != OOM_EXIT_CODE:
            return result.returncode if result.returncode >= 0 else 128 - result.returncode
        if not os.path.isfile(state_path):
            print("OOM before any recovery checkpoint was committed; stopping.", flush=True)
            return OOM_EXIT_CODE
        state = read_state(state_path)
        epoch = state["completed_epoch"]
        stalled = stalled + 1 if committed is not None and epoch <= committed else 0
        if stalled >= 2 or restarts >= max_restarts:
            print("OOM recovery stopped: retry limit reached or no epoch progress. "
                  "Reduce batch_size before trying again.", flush=True)
            return OOM_EXIT_CODE
        committed = epoch
        restarts += 1
        resume = True
        print(f"GPU OOM: worker exited. Restart {restarts}/{max_restarts} "
              f"from saved epoch {epoch}; next epoch {epoch + 1}.", flush=True)
