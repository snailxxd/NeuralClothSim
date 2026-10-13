"""Measure stateful single-frame latency or original full-sequence throughput.

Run from the project root, for example:
  .venv/bin/python scripts/benchmark_inference.py \
      --config ncs/configs/mixamo_resume_207.json --gpu-id 0

Inputs are resident on the selected device. Each timed call includes Python/TF
dispatch, computation, synchronization and copying its outputs to the CPU.
These are wall-clock measurements, NOT isolated GPU kernel times.
Sequence modes include body mesh skinning, as in the original predict() path.
No model/data loading, compilation, file writes, collision correction, normal
updates or rendering are timed.
TF32 is disabled by default so FP32 shape-dependent rounding in forward
kinematics is not amplified by the finite-difference acceleration features.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="Checkpoint directory/prefix; default: checkpoints/<config name>")
    parser.add_argument("--gpu-id", default="0", help="Physical CUDA GPU ID (default: 0)")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu",
                        help="CPU is an explicit fallback for validation only")
    parser.add_argument("--mode", choices=("network", "pipeline", "cached", "cached-xla", "both", "compare", "sequence",
                                          "sequence-graph", "all"), default="both",
                        help="sequence: original predict(); sequence-graph: same path in tf.function; "
                             "cached: single-frame pipeline with device-resident history; "
                             "cached-xla: fused cached pipeline; both: network, pipeline and cached; "
                             "compare: both plus cached-xla; all: also both sequence tests")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300,
                        help="Maximum source output frames, sampled at config FPS")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--motion", type=float, default=1.0)
    parser.add_argument("--tf32", action="store_true",
                        help="Allow TF32 GPU matmuls (off by default); may fail strict validation")
    parser.add_argument("--json", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    for name in ("frames", "warmup", "iterations"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.sequence_index < 0:
        parser.error("--sequence-index must be nonnegative")
    if args.device == "gpu" and (not args.gpu_id.isdigit()):
        parser.error("--gpu-id must identify a single GPU, e.g. 0")
    return args


def main():
    args = parse_args()
    # Configure CUDA before importing TensorFlow or any project modules.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id if args.device == "gpu" else "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(ROOT / "ncs"))
    import numpy as np
    import tensorflow as tf
    from dataset.sequence import PoseSequence
    from global_vars import CHECKPOINTS_DIR, DATA_DIR, TXT_DIR
    from model.ncs import NCS
    from model.streaming import StreamingNCS
    from utils.config import MainConfig

    # Float32 tensors can otherwise use reduced-precision TF32 matmuls on
    # supported GPUs. Batched and three-frame FK may select different kernels;
    # their rounding differences are amplified by acceleration's 1/dt**2.
    tf.config.experimental.enable_tensor_float_32_execution(args.tf32)
    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "gpu" and not gpus:
        raise RuntimeError("No GPU visible to TensorFlow. Run in a GPU-enabled terminal; "
                           "CPU fallback requires --device cpu explicitly.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    device = "/GPU:0" if args.device == "gpu" else "/CPU:0"
    config = MainConfig(str(args.config.resolve()))
    checkpoint = (args.checkpoint.resolve() if args.checkpoint else
                  Path(CHECKPOINTS_DIR) / config.name)
    if not checkpoint.exists() and not Path(str(checkpoint) + ".index").exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Sample just one test sequence; pad its start exactly like Data(mode='test').
    listing = Path(TXT_DIR) / config.data.dataset / config.data.test
    names = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
    if args.sequence_index >= len(names):
        raise ValueError(f"Sequence index out of range: test list contains {len(names)} sequences")
    sequence_path = Path(DATA_DIR) / config.data.dataset / names[args.sequence_index]
    sequence = PoseSequence(str(sequence_path))
    count = min(args.frames, int(np.floor(sequence.duration / config.time_step + 1e-6)) + 1)
    times = np.arange(-2, count, dtype=np.float64) * config.time_step
    poses_np, trans_np = sequence.get(times, extrapolation="clip")

    with tf.device(device):
        model = NCS(config)
        model.build(config.input_shape)
        status = model.load_weights(str(checkpoint))
        status.assert_existing_objects_matched()
        status.expect_partial()  # Training-only optimizer slots are unused.
        network_layers = model.static_encoder + model.dynamic_encoder + model.decoder
        if args.device == "gpu":
            misplaced = [v.name for layer in network_layers for v in layer.weights
                         if "GPU:" not in v.device.upper()]
            if misplaced:
                raise RuntimeError(f"Network weights are not on GPU: {misplaced}")

        joints = model.body.num_input_joints
        units = model.dynamic_encoder[-1].units
        hidden_spec = tf.TensorSpec([1, units], tf.float32)

        @tf.function(input_signature=[tf.TensorSpec([1, 1, joints, 21], tf.float32),
                                      hidden_spec])
        def network_step(features, hidden):
            static, dynamic = tf.split(features, [9, 12], axis=-1)
            for layer in model.static_encoder:
                static = layer(static)
            for layer in model.dynamic_encoder[:-1]:
                dynamic = layer(dynamic)
            dynamic, states = model.dynamic_encoder[-1].cell(
                dynamic[:, 0], [hidden], training=False)
            value = static + args.motion * dynamic[:, None]
            for layer in model.decoder:
                value = layer(value)
            return value, states[0]

        pose_spec = tf.TensorSpec(
            [1, 3, model.body.num_joints, poses_np.shape[-1]], tf.float32)
        trans_spec = tf.TensorSpec([1, 3, 3], tf.float32)

        @tf.function(input_signature=[pose_spec, trans_spec])
        def frame_inputs(poses, trans):
            return model.call_inputs(poses, trans)

        @tf.function(input_signature=[
            pose_spec, trans_spec, hidden_spec])
        def pipeline_step(poses, trans, hidden):
            features, matrices = frame_inputs(poses, trans)
            deformation, hidden = network_step(features, hidden)
            matrices = tf.gather(matrices, model.body.input_joints, axis=-3)
            cloth = model.lbs_cloth(model.garment.vertices + deformation, matrices)
            return cloth, hidden

        @tf.function
        def sequence_graph(poses, trans):
            return model.predict((poses, trans), w=args.motion)

        poses = tf.constant(poses_np[None], dtype=tf.float32)
        trans = tf.constant(trans_np[None], dtype=tf.float32)
        features, _ = model.call_inputs(poses, trans)
        # Prepare individual device-resident inputs outside the timed region.
        feature_frames = [features[:, i:i + 1] for i in range(count)]
        pose_frames = [poses[:, i:i + 3] for i in range(count)]
        trans_frames = [trans[:, i:i + 3] for i in range(count)]
        zero = tf.zeros([1, units], tf.float32)
        stream = StreamingNCS(model, motion=args.motion)
        xla_stream = StreamingNCS(model, motion=args.motion)
        xla_step = tf.function(xla_stream.step.python_function, jit_compile=True)
        current_frames = [(poses[:, i + 2:i + 3], trans[:, i + 2:i + 3])
                          for i in range(count)]
        history = (poses[:, :1], trans[:, :1], poses[:, 1:2], trans[:, 1:2])

        def synchronize(outputs):
            # Reading every output ensures both vertex and hidden computations
            # finish. Keep hidden as a device tensor for the following frame.
            arrays = [tensor.numpy() for tensor in tf.nest.flatten(outputs)]
            if not all(np.isfinite(array).all() for array in arrays):
                raise RuntimeError("Non-finite inference output")

        # Compare both streaming paths against the original sequence prediction.
        # Limit validation memory separately from the benchmark source length.
        check_count = min(count, 32)
        body, reference_cloth, reference_unskinned = model.predict(
            (poses[0, :check_count + 2], trans[0, :check_count + 2]), w=args.motion)
        reference_features, reference_matrices = model.call_inputs(
            poses[:, :check_count + 2], trans[:, :check_count + 2])
        network_hidden = pipeline_hidden = zero
        deformation_error = cloth_error = 0.0
        feature_errors = {"static": 0.0, "velocity": 0.0, "acceleration": 0.0}
        matrix_error = 0.0
        failures = []
        stream.reset(*history).numpy()
        cached_error = 0.0
        for i in range(check_count):
            deformation, network_hidden = network_step(feature_frames[i], network_hidden)
            cloth, pipeline_hidden = pipeline_step(pose_frames[i], trans_frames[i], pipeline_hidden)
            cached_cloth, cached_hidden = stream.step(*current_frames[i])
            cached_error = max(cached_error, float(tf.reduce_max(tf.abs(
                cached_cloth[0] - reference_cloth[i:i + 1])).numpy()))
            if not np.allclose(cached_cloth[0].numpy(), reference_cloth[i:i + 1].numpy(),
                               rtol=1e-4, atol=1e-5):
                failures.append(f"cached frame {i}")
            if not np.allclose(cached_hidden.numpy(), pipeline_hidden.numpy(),
                               rtol=1e-4, atol=1e-5):
                failures.append(f"cached hidden frame {i}")
            expected = reference_unskinned[i:i + 1]
            actual = (model.garment.vertices + deformation)[0]
            if not np.allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-5):
                failures.append(f"network frame {i}")
            if not np.allclose(cloth[0].numpy(), reference_cloth[i:i + 1].numpy(),
                               rtol=1e-4, atol=1e-5):
                failures.append(f"pipeline frame {i}")
            local_features, local_matrices = frame_inputs(pose_frames[i], trans_frames[i])
            diff = tf.abs(local_features - reference_features[:, i:i + 1])
            for name, section in (("static", slice(0, 9)), ("velocity", slice(9, 18)),
                                  ("acceleration", slice(18, 21))):
                feature_errors[name] = max(feature_errors[name],
                                          float(tf.reduce_max(diff[..., section]).numpy()))
            matrix_error = max(matrix_error, float(tf.reduce_max(tf.abs(
                local_matrices - reference_matrices[:, i:i + 1])).numpy()))
            deformation_error = max(deformation_error,
                                    float(tf.reduce_max(tf.abs(actual - expected)).numpy()))
            cloth_error = max(cloth_error,
                              float(tf.reduce_max(tf.abs(cloth[0] - reference_cloth[i:i + 1])).numpy()))
        del body, reference_cloth, reference_unskinned
        # A reset must reproduce the first output after a previous clip's state.
        stream.reset(*history).numpy()
        first_cloth, _ = stream.step(*current_frames[0])
        stream.reset(*history).numpy()
        repeated_cloth, _ = stream.step(*current_frames[0])
        np.testing.assert_allclose(first_cloth.numpy(), repeated_cloth.numpy(), rtol=0, atol=0)
        print(f"Validation diagnostics: TF32={args.tf32}; "
              f"unskinned max error={deformation_error:.6g}; "
              f"cloth max error={cloth_error:.6g}; "
              f"cached cloth max error={cached_error:.6g}; "
              f"FK matrix max error={matrix_error:.6g}; feature errors={feature_errors}",
              flush=True)
        if failures:
            raise RuntimeError(
                f"Streaming validation failed at {', '.join(failures[:6])} "
                "(rtol=1e-4, atol=1e-5); timing was not run. "
                "Compare FK/feature errors above to isolate preprocessing differences. "
                + ("Re-run without --tf32 for full FP32 matmuls." if args.tf32 else
                   "TF32 is already disabled; retain these diagnostics for investigation."))

        xla_error = None
        if args.mode in ("cached-xla", "compare", "all"):
            xla_stream.reset(*history).numpy()
            stream.reset(*history).numpy()
            xla_error = 0.0
            for i in range(check_count):
                actual_cloth, _ = xla_step(*current_frames[i])
                expected_cloth, _ = stream.step(*current_frames[i])
                np.testing.assert_allclose(actual_cloth.numpy(), expected_cloth.numpy(),
                                           rtol=1e-4, atol=1e-5)
                xla_error = max(xla_error, float(tf.reduce_max(tf.abs(
                    actual_cloth - expected_cloth)).numpy()))
            print(f"XLA validation passed; cached path max error={xla_error:.6g}", flush=True)

        def benchmark(mode):
            cached = mode in ("cached", "cached-xla")
            selected_stream = xla_stream if mode == "cached-xla" else stream
            function = (xla_step if mode == "cached-xla" else stream.step if cached else
                        network_step if mode == "network" else pipeline_step)
            inputs = (current_frames if cached else
                      [(x,) for x in feature_frames] if mode == "network" else
                      list(zip(pose_frames, trans_frames)))
            # Finish input preparation before any measurement.
            for tensors in inputs:
                synchronize(tensors)
            hidden = zero
            for i in range(args.warmup):
                index = i % count
                if index == 0:
                    hidden = zero
                    if cached:
                        selected_stream.reset(*history).numpy()
                outputs = function(*inputs[index]) if cached else function(*inputs[index], hidden)
                synchronize(outputs)
                hidden = outputs[1]
            latencies = []
            hidden = zero
            for i in range(args.iterations):
                index = i % count
                if index == 0:
                    hidden = zero  # Restart the source clip, avoiding a jump at its loop boundary.
                    if cached:
                        selected_stream.reset(*history).numpy()
                start = time.perf_counter_ns()
                outputs = function(*inputs[index]) if cached else function(*inputs[index], hidden)
                # Only output readback belongs to the measured interval.
                arrays = [tensor.numpy() for tensor in tf.nest.flatten(outputs)]
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if not all(np.isfinite(array).all() for array in arrays):
                    raise RuntimeError("Non-finite inference output")
                latencies.append(elapsed)
                hidden = outputs[1]
            values = np.asarray(latencies)
            result = {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
                      "p95_ms": float(np.percentile(values, 95)),
                      "p99_ms": float(np.percentile(values, 99)),
                      "min_ms": float(values.min()), "max_ms": float(values.max()),
                      "iterations": args.iterations, "latencies_ms": latencies}
            result["history_cached"] = cached
            result["xla_compiled"] = mode == "cached-xla"
            result["clip_reset_timed"] = False
            print(f"{mode:8s}: median={result['median_ms']:.3f} ms, "
                  f"P95={result['p95_ms']:.3f} ms, P99={result['p99_ms']:.3f} ms, "
                  f"mean={result['mean_ms']:.3f} ms")
            return result

        def benchmark_sequence(mode):
            # Both functions use the original model.predict path, including
            # body LBS and a zero-initialized GRU for each independent clip.
            function = (sequence_graph if mode == "sequence-graph" else
                        lambda p, t: model.predict((p, t), w=args.motion))
            p, t = poses[0], trans[0]
            synchronize((p, t))
            expected = model.predict((p, t), w=args.motion)
            actual = function(p, t)
            max_error = 0.0
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-5)
                max_error = max(max_error, float(tf.reduce_max(tf.abs(a - b)).numpy()))
            del actual, expected
            for _ in range(args.warmup):
                synchronize(function(p, t))
            latencies = []
            for _ in range(args.iterations):
                start = time.perf_counter_ns()
                outputs = function(p, t)
                arrays = [tensor.numpy() for tensor in tf.nest.flatten(outputs)]
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if not all(np.isfinite(array).all() for array in arrays):
                    raise RuntimeError("Non-finite sequence output")
                latencies.append(elapsed)
            values = np.asarray(latencies)
            result = {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
                      "p95_ms": float(np.percentile(values, 95)),
                      "p99_ms": float(np.percentile(values, 99)),
                      "min_ms": float(values.min()), "max_ms": float(values.max()),
                      "iterations": args.iterations, "latencies_ms": latencies,
                      "frames_per_call": count, "includes_body_mesh": True,
                      "outputs": ["body", "cloth", "cloth_unskinned"],
                      "validation_max_abs_error": max_error,
                      "mean_per_frame_ms": float(values.mean() / count),
                      "median_per_frame_ms": float(np.median(values) / count),
                      "fps": float(1000 * count / values.mean())}
            print(f"{mode}: mean clip={result['mean_ms']:.3f} ms / {count} frames; "
                  f"mean/frame={result['mean_per_frame_ms']:.3f} ms; "
                  f"throughput={result['fps']:.1f} FPS; "
                  f"P95 clip={result['p95_ms']:.3f} ms")
            return result

        gpu_details = tf.config.experimental.get_device_details(gpus[0]) if gpus else {}
        report = {
            "tensorflow": tf.__version__, "device": device,
            "gpu_name": gpu_details.get("device_name"),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "config": str(args.config.resolve()), "checkpoint": str(checkpoint),
            "sequence": str(sequence_path), "source_frames": count,
            "model_fps": config.data.fps, "dtype": "float32", "batch_size": 1,
            "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
            "cloth_vertices": model.garment.num_verts, "hidden_units": units,
            "motion": args.motion, "warmup": args.warmup,
            "timing": "wall-clock; device-resident inputs; output readback included",
            "validation": {"frames": check_count,
                           "unskinned_max_abs_error": deformation_error,
                           "cloth_max_abs_error": cloth_error,
                           "cached_cloth_max_abs_error": cached_error,
                           "xla_vs_cached_max_abs_error": xla_error,
                           "fk_matrix_max_abs_error": matrix_error,
                           "feature_max_abs_errors": feature_errors},
            "results": {},
        }
        print(f"Device: {device} {report['gpu_name'] or ''}; TensorFlow {tf.__version__}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Source: {count} frames at {config.data.fps} Hz; batch=1; "
              f"cloth={model.garment.num_verts} vertices; warmup={args.warmup}")
        print(f"Streaming validation passed ({check_count} frames); "
              f"cloth max error={cloth_error:.3g}")
        print("Latency includes dispatch, computation, synchronization and output readback.")
        if args.mode == "all":
            modes = ("network", "pipeline", "cached", "cached-xla", "sequence", "sequence-graph")
        elif args.mode == "compare":
            modes = ("network", "pipeline", "cached", "cached-xla")
        elif args.mode == "both":
            modes = ("network", "pipeline", "cached")
        else:
            modes = (args.mode,)
        print("Sequence times cover the whole clip including body mesh; FPS is throughput, "
              "not single-frame latency.")
        for mode in modes:
            report["results"][mode] = (benchmark_sequence(mode) if mode.startswith("sequence")
                                       else benchmark(mode))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Report: {args.json.resolve()}")


if __name__ == "__main__":
    main()
