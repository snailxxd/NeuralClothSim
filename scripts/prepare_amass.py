"""Convert extracted AMASS SMPL+H motions to NCS SMPL sequences (CPU only)."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]


def convert(source, destination, source_name, target_fps, up_axis):
    with np.load(source, allow_pickle=False) as data:
        poses = np.asarray(data["poses"], dtype=np.float64)
        trans = np.asarray(data["trans"], dtype=np.float64)
        fps = float(data["mocap_framerate"].item())
    if poses.ndim != 2 or poses.shape[1] != 156:
        raise ValueError(f"Expected SMPL+H poses (T, 156), got {poses.shape}")
    if len(poses) < 2 or trans.shape != (len(poses), 3):
        raise ValueError("Need at least two frames and matching (T, 3) translations")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid mocap_framerate: {fps}")
    if not np.isfinite(poses).all() or not np.isfinite(trans).all():
        raise ValueError("Non-finite motion values")

    # SMPL+H shares root + 21 body joints with SMPL. Its remaining joints
    # are fingers, NOT the two terminal SMPL hand joints. Keep those neutral.
    result = np.zeros((len(poses), 24, 3), dtype=np.float64)
    result[:, :22] = poses[:, :66].reshape(-1, 22, 3)
    if up_axis == "z":
        # AMASS world Z-up -> the bundled body's Y-up: (x, y, z) -> (x, z, -y).
        # Only the global root rotation changes; local joint axes stay intact.
        alignment = Rotation.from_euler("x", -90, degrees=True)
        result[:, 0] = (alignment * Rotation.from_rotvec(result[:, 0])).as_rotvec()
        trans = alignment.apply(trans)
    # NCS has a fixed body rooted at zero. Retain displacement and acceleration
    # while removing the source scene's arbitrary absolute position.
    trans -= trans[0].copy()
    if target_fps < fps:
        times = np.arange(len(result), dtype=np.float64) / fps
        samples = np.arange(int(np.floor(times[-1] * target_fps)) + 1) / target_fps
        if len(samples) < 2:
            raise ValueError("Motion is too short at the requested output FPS")
        resampled = np.zeros((len(samples), 24, 3), dtype=np.float64)
        for joint in range(22):
            resampled[:, joint] = Slerp(
                times, Rotation.from_rotvec(result[:, joint])
            )(samples).as_rotvec()
        trans = np.stack([np.interp(samples, times, trans[:, i]) for i in range(3)], axis=-1)
        result, fps = resampled, target_fps
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, poses=result.astype(np.float32),
                        trans=trans.astype(np.float32), fps=np.float64(fps),
                        source=source_name)
    return len(result), fps


def split_subjects(names, seed):
    subjects = sorted({Path(name).parent.as_posix() for name in names})
    if len(subjects) < 3:
        raise ValueError("Need at least three subject directories for disjoint train/val/test")
    np.random.default_rng(seed).shuffle(subjects)
    held_out = max(1, int(len(subjects) * 0.1))
    validation = set(subjects[:held_out])
    test = set(subjects[held_out:2 * held_out])
    splits = {"train": [], "val": [], "test": []}
    for name in sorted(names):
        subject = Path(name).parent.as_posix()
        key = "val" if subject in validation else "test" if subject in test else "train"
        splits[key].append(name)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Extracted CMU folder containing subject directories")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "amass_cmu")
    parser.add_argument("--lists", type=Path, default=ROOT / "ncs/dataset/txt/amass_cmu")
    parser.add_argument("--fps", type=float, default=30,
                        help="Downsample to this FPS; slower sources keep their original FPS")
    parser.add_argument("--source-up-axis", choices=("z", "y"), default="z")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    source = args.source.resolve()
    if not source.is_dir():
        parser.error(f"Source directory does not exist: {source}")
    files = sorted(source.rglob("*_poses.npz"))
    if not files:
        parser.error("No *_poses.npz files found; extract the AMASS CMU archive first")
    names = [p.relative_to(source).as_posix() for p in files]
    if any("\n" in n or "\r" in n for n in names):
        parser.error("Sequence paths cannot contain newlines")
    try:
        splits = split_subjects(names, args.seed)
    except ValueError as error:
        parser.error(str(error))
    # Refuse to mix an old conversion with different preprocessing or splits.
    for destination in (args.output, args.lists):
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            parser.error(f"Destination must be empty: {destination}")
    frames, seconds = 0, 0.0
    for index, (path, name) in enumerate(zip(files, names), 1):
        try:
            count, fps = convert(path, args.output / name, name, args.fps, args.source_up_axis)
        except (ValueError, KeyError, OSError) as error:
            raise RuntimeError(f"Cannot convert {path}: {error}. Remove partial output before retrying.") from error
        frames += count
        seconds += (count - 1) / fps
        if index % 100 == 0 or index == len(files):
            print(f"Converted {index}/{len(files)} sequences", flush=True)
    args.lists.mkdir(parents=True, exist_ok=True)
    for key, entries in splits.items():
        (args.lists / f"{key}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
        print(f"{key}: {len(entries)} sequences")
    metadata = dict(source=str(source), source_up_axis=args.source_up_axis,
                    target_fps=args.fps, seed=args.seed, frames=frames,
                    duration_seconds=seconds, split_unit="subject_directory",
                    smpl_terminal_hands="zero", shape="fixed bundled body; betas/dmpls ignored")
    (args.lists / "preprocessing.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Done: {frames} frames, {seconds / 3600:.2f} hours -> {args.output}")


if __name__ == "__main__":
    main()
