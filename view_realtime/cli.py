"""Fixed-step model inference with smooth GPU animation rendering.

Playback follows wall time; model ticks and display FPS are independent.
"""

import argparse
from collections import deque, namedtuple
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
InferenceFrame = namedtuple('InferenceFrame', 'local matrices started finished reset_ms')


from .playback import PlaybackClock, FixedStepPlayback, DisplaySkeleton, sample_interpolated_motion


def validate_weight_restore(status):
    """Validate TensorFlow checkpoints; HDF5 weight loads return no status."""
    if status is None:
        return
    status.assert_existing_objects_matched()
    status.expect_partial()


def parse_args():
    parser = argparse.ArgumentParser(prog='python -m view_realtime', description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'ncs/configs/test.json')
    parser.add_argument('--checkpoint', type=Path,
                        help='Weights to load; defaults to the validation-best weights')
    parser.add_argument('--gpu-id', default='0')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--sequence-index', type=int, default=0)
    parser.add_argument('--motion', type=float, default=1.0)
    parser.add_argument('--vsync', action=argparse.BooleanOptionalAction, default=False,
                        help='Limit drawing to display refresh; independent of animation time')
    parser.add_argument('--ui', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--speed', type=float, default=1.0, help='Animation playback multiplier (0.1 to 4)')
    parser.add_argument('--xla', action=argparse.BooleanOptionalAction, default=True,
                        help='Use XLA; fall back to ordinary inference if validation fails')
    parser.add_argument('--render-device', choices=['hardware', 'auto', 'software'], default='hardware')
    parser.add_argument('--render-adapter', help='WSL D3D12 adapter name substring; defaults to inference GPU')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--duration', type=float,
                        help='Optional session limit in seconds, excluding startup/warmup')
    parser.add_argument('--json', type=Path)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate streaming outputs without creating a window')
    args = parser.parse_args()
    for key in ('width', 'height'):
        if getattr(args, key) <= 0:
            parser.error(f'--{key} must be positive')
    if args.warmup < 0 or args.sequence_index < 0:
        parser.error('--warmup and --sequence-index must be nonnegative')
    if not args.gpu_id.isdigit():
        parser.error('--gpu-id must identify one GPU')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be finite and positive')
    if not math.isfinite(args.speed) or not 0.1 <= args.speed <= 4:
        parser.error('--speed must be between 0.1 and 4')
    return args


def validate(tf, np, model, stream, step, poses, trans, count, reset):
    """Check full outputs, old API compatibility, recurrence and reset behavior."""
    reference = model.predict((tf.constant(poses[:count + 2]),
                               tf.constant(trans[:count + 2])), w=stream.motion)
    reset()
    body_error = cloth_error = 0.0
    first = None
    for index in range(count):
        p = tf.constant(poses[None, index + 2:index + 3])
        t = tf.constant(trans[None, index + 2:index + 3])
        body, cloth, hidden = step(p, t)
        body, cloth, hidden = body.numpy(), cloth.numpy(), hidden.numpy()
        if first is None:
            first = (body.copy(), cloth.copy(), hidden.copy())
        for actual, expected in ((body[0, 0], reference[0][index].numpy()),
                                 (cloth[0, 0], reference[1][index].numpy())):
            # Batched prediction and single-frame inference use different
            # float32 kernels; allow small coordinate drift near zero too.
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3,
                                       equal_nan=False,
                                       err_msg=f'Batched/streaming mismatch at frame {index}')
        body_error = max(body_error, float(np.max(np.abs(body[0, 0] - reference[0][index]))))
        cloth_error = max(cloth_error, float(np.max(np.abs(cloth[0, 0] - reference[1][index]))))
    reset()
    p, t = tf.constant(poses[None, 2:3]), tf.constant(trans[None, 2:3])
    repeated = tuple(value.numpy() for value in step(p, t))
    for actual, expected in zip(repeated, first):
        np.testing.assert_array_equal(actual, expected)
    # Compare old and new APIs over consecutive frames, including GRU state.
    reset()
    old_outputs = []
    for index in range(count):
        output = stream.step(tf.constant(poses[None, index + 2:index + 3]),
                             tf.constant(trans[None, index + 2:index + 3]))
        old_outputs.append(tuple(value.numpy() for value in output))
    reset()
    for index, expected in enumerate(old_outputs):
        _, cloth, hidden = step(tf.constant(poses[None, index + 2:index + 3]),
                                tf.constant(trans[None, index + 2:index + 3]))
        for actual, value in zip((cloth.numpy(), hidden.numpy()), expected):
            np.testing.assert_allclose(actual, value, rtol=1e-4, atol=1e-4,
                                       equal_nan=False,
                                       err_msg=f'Old/local streaming API mismatch at frame {index}')
    reset()
    return {'frames': count, 'coordinate_atol': 1e-3, 'rtol': 1e-4,
            'body_max_abs_error': body_error,
            'cloth_max_abs_error': cloth_error, 'old_api_and_reset': 'passed'}, first[:2]


def save_report(args, report, samples, np):
    samples = list(samples)
    summary = {}
    if samples:
        for key in samples[0]:
            if not key.endswith('_ms') or key == 'inference_samples_ms':
                continue
            values = np.asarray([value for row in samples
                                 for value in (row.get('inference_samples_ms', [row[key]])
                                               if key == 'inference_readback_ms' else [row[key]])
                                 if value is not None])
            if len(values):
                summary[key] = {'mean': float(values.mean()),
                                'p50': float(np.percentile(values, 50)),
                                'p95': float(np.percentile(values, 95)),
                                'p99': float(np.percentile(values, 99))}
        total = np.array([row['processing_ms'] for row in samples])
        summary['processing_fps'] = float(1000 / total.mean())
        summary['over_budget_fraction'] = float(np.mean(total > 1000 / report['model_fps']))
        if report.get('session_wall_seconds', 0) > 0:
            summary['observed_draws_per_second'] = report.get('total_frames', len(samples)) / report['session_wall_seconds']
            summary['observed_model_steps_per_second'] = report.get('simulation_steps', 0) / report['session_wall_seconds']
        if 'inference_readback_ms' in summary:
            summary['serial_model_steps_per_second_estimate'] = 1000 / summary['inference_readback_ms']['mean']
    report.update(summary=summary, measured_frames=report.get('total_frames', len(samples)),
                  retained_samples=len(samples), samples=samples)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    if args.csv and samples:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
    print(json.dumps({key: value for key, value in report.items() if key != 'samples'}, indent=2))


def memory_sample(tf, device, elapsed):
    result = {'elapsed_seconds': elapsed}
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                result['process_rss_mib'] = int(line.split()[1]) / 1024
                break
    if device == '/GPU:0':
        result['tensorflow_allocator_bytes'] = tf.config.experimental.get_memory_info('GPU:0')
    return result


def main():
    args = parse_args()
    # The dynamic loader reads LD_LIBRARY_PATH at process startup. Re-exec once
    # so WSL Mesa can find D3D12 and DXCore without editing system configuration.
    wsl_lib = '/usr/lib/wsl/lib'
    if not args.validate_only and Path(wsl_lib, 'libd3d12.so').exists():
        paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
        if wsl_lib not in paths:
            os.environ['LD_LIBRARY_PATH'] = ':'.join([wsl_lib] + [p for p in paths if p])
            os.execv(sys.executable, [sys.executable, '-m', 'view_realtime', *sys.argv[1:]])
        if args.render_device != 'software':
            os.environ.setdefault('GALLIUM_DRIVER', 'd3d12')
    if args.render_device == 'software':
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id if args.device == 'gpu' else '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    sys.path.insert(0, str(ROOT / 'ncs'))
    import numpy as np
    import tensorflow as tf
    from dataset.sequence import PoseSequence
    from global_vars import CHECKPOINTS_DIR, DATA_DIR, TXT_DIR
    from model.ncs import NCS
    from model.streaming import StreamingNCS
    from utils.config import MainConfig

    tf.config.experimental.enable_tensor_float_32_execution(False)
    gpus = tf.config.list_physical_devices('GPU')
    if args.device == 'gpu' and not gpus:
        raise RuntimeError('No TensorFlow GPU. Use a GPU-enabled terminal or --device cpu.')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    config = MainConfig(str(args.config.resolve()))
    checkpoint = (args.checkpoint.resolve() if args.checkpoint else
                  Path(CHECKPOINTS_DIR) / config.name / 'best.weights.h5')
    if not checkpoint.exists() and not Path(str(checkpoint) + '.index').exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    names = [line.strip() for line in (Path(TXT_DIR) / config.data.dataset /
                                      config.data.test).read_text().splitlines() if line.strip()]
    if args.sequence_index >= len(names):
        raise ValueError(f'Sequence index out of range: {len(names)} test sequences')
    source = Path(DATA_DIR) / config.data.dataset / names[args.sequence_index]
    sequence = PoseSequence(str(source))
    if sequence.poses.shape[-1] != 4:
        raise ValueError('Viewer requires quaternion motion poses for SLERP playback')
    count = int(np.floor(sequence.duration / config.time_step + 1e-6)) + 1
    poses, trans = sequence.get(np.arange(-2, count) * config.time_step, extrapolation='clip')
    device = '/GPU:0' if args.device == 'gpu' else '/CPU:0'
    samples, pending = deque(maxlen=10000), deque()
    renderer = None
    with tf.device(device):
        start = time.perf_counter()
        model = NCS(config)
        model.build(config.input_shape)
        status = model.load_weights(str(checkpoint))
        validate_weight_restore(status)
        stream = StreamingNCS(model, motion=args.motion)
        base_step = stream.step_local
        step = tf.function(base_step.python_function, jit_compile=True) if args.xla else base_step

        def validation_step(p, t):
            local, matrices, hidden = step(p, t)
            body = model.lbs_body(model.body.vertices, matrices)
            cloth = model.lbs_cloth(local, tf.gather(matrices, model.body.input_joints, axis=-3))
            return body, cloth, hidden
        history = (tf.constant(poses[None, :1]), tf.constant(trans[None, :1]),
                   tf.constant(poses[None, 1:2]), tf.constant(trans[None, 1:2]))

        initial_cache = None
        state_variables = (stream.previous_x, stream.previous_j, stream.older_j,
                           stream.hidden, stream.initialized)

        @tf.function
        def restore_initial_cache():
            for variable, value in zip(state_variables, initial_cache):
                variable.assign(value)
            return stream.initialized.read_value()

        def reset():
            nonlocal initial_cache
            if initial_cache is None:
                stream.reset(*history).numpy()
                initial_cache = tuple(variable.read_value() for variable in state_variables)
            else:
                restore_initial_cache().numpy()

        load_ms = (time.perf_counter() - start) * 1000
        reset()
        start = time.perf_counter()
        for tensor in step(tf.constant(poses[None, 2:3]), tf.constant(trans[None, 2:3])):
            tensor.numpy()
        first_ms = (time.perf_counter() - start) * 1000
        xla_enabled = args.xla
        xla_fallback_reason = None
        try:
            validation, first = validate(tf, np, model, stream, validation_step, poses, trans,
                                         min(count, 32), reset)
        except AssertionError as error:
            if not xla_enabled:
                raise
            xla_fallback_reason = str(error)
            print('XLA numerical validation failed; retrying without XLA.',
                  file=sys.stderr, flush=True)
            step = base_step
            xla_enabled = False
            # validate() resets cached features and GRU state before stepping.
            validation, first = validate(tf, np, model, stream, validation_step, poses, trans,
                                         min(count, 32), reset)
        print(f'Validation passed: {validation}', flush=True)
        report = {'config': str(args.config.resolve()), 'checkpoint': str(checkpoint),
                  'sequence': str(source), 'source_frames': count, 'model_fps': config.data.fps,
                  'tensorflow': tf.__version__, 'device': device, 'xla': xla_enabled,
                  'xla_requested': args.xla, 'xla_fallback_reason': xla_fallback_reason,
                  'gpu': tf.config.experimental.get_device_details(gpus[0]) if gpus else {},
                  'mode': 'realtime', 'tf32': False, 'body_vertices': model.body.num_verts,
                  'cloth_vertices': model.garment.num_verts, 'motion': args.motion,
                  'warmup_frames': args.warmup, 'model_load_ms': load_ms,
                  'first_step_including_compile_ms': first_ms, 'validation': validation}
        # Device details may contain a tuple; JSON handles it as an array.
        if args.validate_only:
            save_report(args, report, samples, np)
            return
        from .realtime_renderer import Renderer, is_software_renderer
        if Path(wsl_lib, 'libd3d12.so').exists():
            adapter = args.render_adapter or report['gpu'].get('device_name')
            if adapter:
                os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME'] = adapter
        # Prepare the source pose tensors once: per-frame host upload/readback
        # was unnecessary for playback of an already loaded motion sequence.
        pose_tensor, trans_tensor = tf.constant(poses[None]), tf.constant(trans[None])
        device_frames = [(pose_tensor[:, i + 2:i + 3], trans_tensor[:, i + 2:i + 3])
                         for i in range(count)]
        for frame in device_frames:
            for value in frame:
                value.numpy()

        skeleton = DisplaySkeleton(model.body)
        # Reuse trained (including trainable softmax) cloth weights in OpenGL.
        weights = model.lbs_cloth.blend_weights
        weights = tf.nn.softmax(weights) if model.lbs_cloth.trainable else weights
        cloth_weights = np.zeros((model.garment.num_verts, model.body.num_joints), dtype='f4')
        cloth_weights[:, model.body.input_joints] = weights.numpy()
        report.update(inputs_device_resident=True, normals='gpu',
                      display_interpolation='local cloth linear / pose SLERP then FK',
                      display_delay_animation_ms=config.time_step * 1000,
                      body_skinning='opengl', cloth_skinning='opengl',
                      render_device_requested=args.render_device,
                      retained_sample_limit=samples.maxlen)
        try:
            start = time.perf_counter()
            renderer = Renderer(first[0][0, 0], first[1][0, 0], model.body.faces,
                                model.garment.faces, args.width, args.height, vsync=args.vsync,
                                require_hardware=args.render_device == 'hardware',
                                ui=args.ui, playback_speed=args.speed, motion=args.motion,
                                body_rest=model.body.vertices,
                                body_weights=model.lbs_body.blend_weights.numpy(),
                                cloth_weights=cloth_weights)
            renderer_init_ms = (time.perf_counter() - start) * 1000
            # Verify local-output GPU skinning against the original TensorFlow
            # model for both meshes, and independent display FK at keyframes.
            reset()
            body_error = cloth_error = fk_error = 0.0
            for i, (p, t) in enumerate(device_frames[:32]):
                local, matrices, _ = step(p, t)
                expected_body = model.lbs_body(model.body.vertices, matrices).numpy()[0, 0]
                expected_cloth = model.lbs_cloth(local, tf.gather(matrices, model.body.input_joints, axis=-3)).numpy()[0, 0]
                display_matrices = skeleton.matrices(poses[i + 2], trans[i + 2])
                np.testing.assert_allclose(display_matrices, matrices.numpy()[0, 0], rtol=1e-4, atol=1e-5)
                fk_error = max(fk_error, float(np.max(np.abs(display_matrices - matrices.numpy()[0, 0]))))
                renderer.upload_local_pair(local.numpy(), local.numpy())
                renderer.upload_display(display_matrices, 0)
                renderer.update_mesh()
                for mesh, expected, label in zip(renderer.meshes, (expected_body, expected_cloth), ('body', 'cloth')):
                    actual = np.frombuffer(mesh[0].read(), dtype='f4').reshape(-1, 6)[:, :3]
                    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                    error = float(np.max(np.abs(actual - expected)))
                    if label == 'body':
                        body_error = max(body_error, error)
                    else:
                        cloth_error = max(cloth_error, error)
            report['validation'].update(opengl_body_max_abs_error=body_error,
                                        opengl_cloth_max_abs_error=cloth_error,
                                        display_fk_max_abs_error=fk_error)
            info = renderer.ctx.info
            report.update(opengl={key: info.get(key) for key in ('GL_VENDOR', 'GL_RENDERER', 'GL_VERSION')},
                          software_renderer=is_software_renderer(info.get('GL_RENDERER', '')),
                          window_size=[args.width, args.height], vsync_requested=args.vsync,
                          ui_enabled=args.ui, playback_speed_initial=args.speed,
                          renderer_init_ms=renderer_init_ms,
                          timing_notes='Inference: cached fixed-step features/network/local cloth readback; '
                          'no TensorFlow cloth skinning. Inference samples exclude frames without model ticks. '
                          'Display: CPU pose SLERP/FK; GPU local cloth interpolation, body/cloth LBS and normals. '
                          'Processing excludes UI build and delayed query collection. CPU/GPU stages overlap; '
                          'swap completion is not screen presentation. Realtime draw FPS is not model throughput.')
            print(f'OpenGL renderer: {info.get("GL_RENDERER")}', flush=True)

            def compute_frame(index):
                with tf.device(device):
                    reset_started = time.perf_counter()
                    if index == 0:
                        reset()
                    infer_started = time.perf_counter()
                    local, matrices, _ = step(*device_frames[index])
                    local, matrices = local.numpy()[0, 0], matrices.numpy()[0, 0]
                    if not (np.isfinite(local).all() and np.isfinite(matrices).all()):
                        raise RuntimeError('Non-finite inference result')
                    return InferenceFrame(local, matrices, infer_started, time.perf_counter(),
                                          (infer_started - reset_started) * 1000)

            for index in range(args.warmup):
                frame = compute_frame(index % count)
                renderer.upload_local_pair(frame.local, frame.local)
                renderer.upload_display(frame.matrices, 0)
                renderer.update_mesh()
                renderer.draw()
                renderer.swap()
                renderer.glfw.poll_events()
                if renderer.glfw.window_should_close(renderer.window):
                    break
            renderer.ctx.finish()
            renderer.frame_times.clear()
            renderer.fps = None
            reset()
            last_frame = None
            started = last_memory = time.perf_counter()
            clock = PlaybackClock(started)
            playback = FixedStepPlayback(count, config.time_step, compute_frame)
            previous_pair = None
            report.update(memory_samples=[memory_sample(tf, device, 0)], completed=False,
                          total_frames=0, simulation_steps=0)
            applied_motion = args.motion
            while not renderer.glfw.window_should_close(renderer.window):
                event_started = time.perf_counter()
                renderer.glfw.poll_events()
                if renderer.ui:
                    renderer.ui.state.update(animation_time=clock.time,
                                             model_fps=1 / config.time_step,
                                             display_delay_ms=report['display_delay_animation_ms'])
                renderer.begin_ui()
                if renderer.motion != applied_motion:
                    stream.motion.assign(renderer.motion)
                    applied_motion = renderer.motion
                event_done = time.perf_counter()
                if args.duration is not None and event_done - started >= args.duration:
                    report.update(completed=True, stop_reason='duration')
                    break
                if renderer.restart:
                    reset()
                    previous_pair = None
                    clock = PlaybackClock(event_done)
                    playback = FixedStepPlayback(count, config.time_step, compute_frame)
                    renderer.restart = False
                step_amount = (config.time_step + max(0, config.time_step - clock.time)
                               if renderer.single_step else 0)
                animation_time = clock.tick(event_done, renderer.paused, renderer.playback_speed, step_amount)
                frame_started = time.perf_counter()
                previous, current, alpha, source_time, computed = playback.sample(animation_time)
                inference_done = time.perf_counter()
                pair = (id(previous), id(current))
                if pair != previous_pair:
                    renderer.upload_local_pair(previous.local, current.local)
                    previous_pair = pair
                pair_uploaded = time.perf_counter()
                pose, translation = sample_interpolated_motion(sequence, source_time)
                matrices = skeleton.matrices(pose, translation)
                pose_done = time.perf_counter()
                renderer.upload_display(matrices, alpha)
                upload_done = time.perf_counter()
                normal_query = renderer.update_mesh(measure=True)
                normal_submit_done = time.perf_counter()
                query = renderer.draw(measure=True)
                draw_done = time.perf_counter()
                renderer.swap()
                finished = time.perf_counter()
                model_times = [(f.finished - f.started) * 1000 for f in computed]
                row = {'sample': report['total_frames'], 'clip_frame': playback.index,
                       'steps_advanced': len(computed), 'animation_time_s': animation_time,
                       'display_time_s': max(0, animation_time - config.time_step),
                       'source_time_s': source_time, 'interpolation_alpha': alpha,
                       'playback_speed': renderer.playback_speed, 'motion': renderer.motion,
                       'vsync_requested': renderer.vsync,
                       'width': renderer.ctx.viewport[2], 'height': renderer.ctx.viewport[3],
                       'event_poll_ms': (event_done - event_started) * 1000,
                       'query_collect_ms': 0.0,
                       'inference_readback_ms': float(np.mean(model_times)) if model_times else None,
                       'inference_samples_ms': model_times,
                       'inference_total_ms': sum(model_times),
                       'inference_wait_ms': (inference_done - frame_started) * 1000,
                       'clip_reset_ms': sum(f.reset_ms for f in computed),
                       'display_pose_ms': (pose_done - pair_uploaded) * 1000,
                       'vertex_upload_cpu_ms': (pair_uploaded - inference_done + upload_done - pose_done) * 1000,
                       'mesh_update_submit_cpu_ms': (normal_submit_done - upload_done) * 1000,
                       'draw_submit_cpu_ms': (draw_done - normal_submit_done) * 1000,
                       'swap_ms': (finished - draw_done) * 1000,
                       'processing_ms': (finished - frame_started) * 1000,
                       'update_interval_ms': None if last_frame is None else (finished - last_frame) * 1000,
                       'gpu_draw_ms': None, 'gpu_mesh_update_ms': None}
                samples.append(row)
                report['total_frames'] += 1
                report['simulation_steps'] += len(computed)
                if query is not None:
                    pending.append((row, query, normal_query))
                collect_started = time.perf_counter()
                if len(pending) > 4:
                    old, old_query, old_normal = pending.popleft()
                    old['gpu_draw_ms'] = old_query.elapsed / 1e6
                    old['gpu_mesh_update_ms'] = old_normal.elapsed / 1e6
                row['query_collect_ms'] = (time.perf_counter() - collect_started) * 1000
                if renderer.ui:
                    renderer.ui.history.append(row)
                last_frame = finished
                renderer.single_step = False
                if renderer.paused:
                    renderer.glfw.wait_events_timeout(0.005)
                if finished - last_memory >= 30:
                    report['memory_samples'].append(memory_sample(tf, device, finished - started))
                    last_memory = finished
            report['session_wall_seconds'] = time.perf_counter() - started
            report.update(playback_speed_final=renderer.playback_speed, vsync_final=renderer.vsync,
                          motion_final=renderer.motion,
                          animation_time_s=clock.time,
                          rendered_frames=report['total_frames'],
                          render_fps=report['total_frames'] / report['session_wall_seconds'],
                          simulation_updates_per_second=report['simulation_steps'] / report['session_wall_seconds'])
            report['memory_samples'].append(memory_sample(tf, device, report['session_wall_seconds']))
        finally:
            if renderer:
                try:
                    while pending:
                        row, query, normal_query = pending.popleft()
                        row['gpu_draw_ms'] = query.elapsed / 1e6
                        row['gpu_mesh_update_ms'] = normal_query.elapsed / 1e6
                    save_report(args, report, samples, np)
                finally:
                    renderer.close()


