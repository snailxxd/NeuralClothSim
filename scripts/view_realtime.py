"""Infer and draw every motion frame with ModernGL + GLFW.

Run from any directory. Model time always advances by config.time_step;
benchmark mode removes pacing, not temporal history.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


class FramePrefetch:
    """One serial inference worker, at most one frame ahead of the renderer.

    The worker never touches OpenGL. Drain before replay/reset so discarded
    predictions cannot overwrite the new clip's recurrent state.
    """

    def __init__(self, compute):
        self.compute = compute
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ncs-inference')
        self.future = None
        self.index = None

    def prefetch(self, index):
        if self.future is not None:
            raise RuntimeError('Only one prefetched frame is allowed')
        self.index = index
        self.future = self.executor.submit(self.compute, index)

    def take(self, index):
        if self.future is None:
            self.prefetch(index)
        if self.index != index:
            raise RuntimeError('Prefetched frame is out of order')
        future, self.future = self.future, None
        return future.result()

    def discard(self):
        if self.future is not None:
            future, self.future = self.future, None
            future.result()

    def close(self):
        try:
            self.discard()
        finally:
            self.executor.shutdown(wait=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'ncs/configs/mixamo_resume_2778.json')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--gpu-id', default='0')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--sequence-index', type=int, default=0)
    parser.add_argument('--motion', type=float, default=1.0)
    parser.add_argument('--mode', choices=['realtime', 'benchmark'], default='benchmark')
    parser.add_argument('--xla', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pipeline', action=argparse.BooleanOptionalAction, default=True,
                        help='Overlap next-frame inference with rendering in benchmark mode')
    parser.add_argument('--render-device', choices=['hardware', 'auto', 'software'], default='hardware')
    parser.add_argument('--render-adapter', help='WSL D3D12 adapter name substring; defaults to inference GPU')
    parser.add_argument('--normals', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--body-skinning', choices=['auto', 'tensorflow'], default='auto',
                        help='auto: skin body in OpenGL when GPU normals are selected')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=0,
                        help='Measured frames in benchmark mode; 0 means unlimited (default)')
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
    if args.warmup < 0 or args.sequence_index < 0 or args.iterations < 0:
        parser.error('--warmup, --iterations and --sequence-index must be nonnegative')
    if not args.gpu_id.isdigit():
        parser.error('--gpu-id must identify one GPU')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be finite and positive')
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
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
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
            np.testing.assert_allclose(actual, value, rtol=1e-4, atol=1e-5)
    reset()
    return {'frames': count, 'body_max_abs_error': body_error,
            'cloth_max_abs_error': cloth_error, 'old_api_and_reset': 'passed'}, first[:2]


def save_report(args, report, samples, np):
    samples = list(samples)
    summary = {}
    if samples:
        for key in samples[0]:
            if not key.endswith('_ms'):
                continue
            values = np.asarray([row[key] for row in samples if row[key] is not None])
            if len(values):
                summary[key] = {'mean': float(values.mean()),
                                'p50': float(np.percentile(values, 50)),
                                'p95': float(np.percentile(values, 95)),
                                'p99': float(np.percentile(values, 99))}
        total = np.array([row['processing_ms'] for row in samples])
        summary['processing_fps'] = float(1000 / total.mean())
        summary['over_budget_fraction'] = float(np.mean(total > 1000 / report['model_fps']))
        if report.get('session_wall_seconds', 0) > 0:
            summary['observed_updates_per_second'] = report.get('total_frames', len(samples)) / report['session_wall_seconds']
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
            os.execv(sys.executable, [sys.executable, *sys.argv])
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
    checkpoint = args.checkpoint.resolve() if args.checkpoint else Path(CHECKPOINTS_DIR) / config.name
    if not checkpoint.exists() and not Path(str(checkpoint) + '.index').exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    names = [line.strip() for line in (Path(TXT_DIR) / config.data.dataset /
                                      config.data.test).read_text().splitlines() if line.strip()]
    if args.sequence_index >= len(names):
        raise ValueError(f'Sequence index out of range: {len(names)} test sequences')
    source = Path(DATA_DIR) / config.data.dataset / names[args.sequence_index]
    sequence = PoseSequence(str(source))
    count = int(np.floor(sequence.duration / config.time_step + 1e-6)) + 1
    poses, trans = sequence.get(np.arange(-2, count) * config.time_step, extrapolation='clip')
    device = '/GPU:0' if args.device == 'gpu' else '/CPU:0'
    samples, pending = deque(maxlen=args.iterations or 10000), deque()
    renderer = pipeline = None
    with tf.device(device):
        start = time.perf_counter()
        model = NCS(config)
        model.build(config.input_shape)
        status = model.load_weights(str(checkpoint))
        status.assert_existing_objects_matched()
        status.expect_partial()
        stream = StreamingNCS(model, motion=args.motion)
        gpu_body = args.normals == 'gpu' and args.body_skinning == 'auto' and not args.validate_only
        base_step = stream.step_for_render if gpu_body else stream.step_with_body
        step = tf.function(base_step.python_function, jit_compile=True) if args.xla else base_step

        def validation_step(p, t):
            if gpu_body:
                cloth, matrices, hidden = step(p, t)
                return model.lbs_body(model.body.vertices, matrices), cloth, hidden
            return step(p, t)
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
        validation, first = validate(tf, np, model, stream, validation_step, poses, trans,
                                     min(count, 32), reset)
        print(f'Validation passed: {validation}', flush=True)
        report = {'config': str(args.config.resolve()), 'checkpoint': str(checkpoint),
                  'sequence': str(source), 'source_frames': count, 'model_fps': config.data.fps,
                  'tensorflow': tf.__version__, 'device': device, 'xla': args.xla,
                  'gpu': tf.config.experimental.get_device_details(gpus[0]) if gpus else {},
                  'mode': args.mode, 'tf32': False, 'body_vertices': model.body.num_verts,
                  'cloth_vertices': model.garment.num_verts, 'motion': args.motion,
                  'warmup_frames': args.warmup, 'model_load_ms': load_ms,
                  'first_step_including_compile_ms': first_ms, 'validation': validation}
        # Device details may contain a tuple; JSON handles it as an array.
        if args.validate_only:
            save_report(args, report, samples, np)
            return
        from realtime_renderer import Renderer, vertex_normals, is_software_renderer
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
        report.update(inputs_device_resident=True, normals=args.normals,
                      pipeline=args.pipeline and args.mode == 'benchmark',
                      clip_reset='restore device-resident initial cache',
                      body_skinning='opengl' if gpu_body else 'tensorflow',
                      render_device_requested=args.render_device,
                      retained_sample_limit=samples.maxlen)
        try:
            start = time.perf_counter()
            renderer = Renderer(first[0][0, 0], first[1][0, 0], model.body.faces,
                                model.garment.faces, args.width, args.height,
                                vsync=args.mode == 'realtime', normals=args.normals,
                                require_hardware=args.render_device == 'hardware',
                                body_rest=model.body.vertices if gpu_body else None,
                                body_weights=model.lbs_body.blend_weights.numpy() if gpu_body else None)
            renderer_init_ms = (time.perf_counter() - start) * 1000
            # Validate actual OpenGL body skinning for every reference frame;
            # this readback is startup-only, never part of the timed frame loop.
            if gpu_body:
                reset()
                max_body_error = 0.0
                for p, t in device_frames[:32]:
                    cloth, matrices, _ = step(p, t)
                    expected_body = model.lbs_body(model.body.vertices, matrices).numpy()[0, 0]
                    renderer.upload(None, cloth.numpy()[0, 0], matrices=matrices.numpy())
                    renderer.update_normals()
                    actual = np.frombuffer(renderer.meshes[0][0].read(), dtype='f4').reshape(-1, 6)[:, :3]
                    np.testing.assert_allclose(actual, expected_body, rtol=1e-4, atol=1e-5)
                    max_body_error = max(max_body_error, float(np.max(np.abs(actual - expected_body))))
                report['validation']['opengl_body_max_abs_error'] = max_body_error
            info = renderer.ctx.info
            renderer_name = info.get('GL_RENDERER', 'unknown')
            software = is_software_renderer(renderer_name)
            report.update(opengl={key: info.get(key) for key in
                                  ('GL_VENDOR', 'GL_RENDERER', 'GL_VERSION')},
                          software_renderer=software, window_size=[args.width, args.height],
                          vsync_requested=args.mode == 'realtime',
                          renderer_init_ms=renderer_init_ms,
                          timing_notes='Wall clock; source inputs device resident; '
                          'inference reads cloth+bone matrices (OpenGL body) or body+cloth (TF body). '
                          'GPU mesh query covers skinning/normals; draw query covers clear/draw/HUD, '
                          'not upload or presentation; worker/render stages overlap when pipeline=true. '
                          'Processing includes waiting for inference/reset, excludes pacing '
                          'and delayed query collection. Swap is not screen presentation.')
            print(f'OpenGL renderer: {renderer_name}; software={software}', flush=True)
            # Warm up the complete path, including buffer updates and drawing.
            for index in range(args.warmup):
                if index % count == 0:
                    reset()
                output = step(*device_frames[index % count])
                if gpu_body:
                    cloth, matrices = output[0].numpy()[0, 0], output[1].numpy()
                    body = None
                else:
                    body, cloth = output[0].numpy()[0, 0], output[1].numpy()[0, 0]
                    matrices = None
                bn = vertex_normals(body, model.body.faces) if args.normals == 'cpu' else None
                cn = vertex_normals(cloth, model.garment.faces) if args.normals == 'cpu' else None
                renderer.upload(body, cloth, bn, cn, matrices=matrices)
                renderer.update_normals()
                renderer.draw()
                renderer.swap()
                renderer.glfw.poll_events()
                if renderer.glfw.window_should_close(renderer.window):
                    break
            renderer.ctx.finish()
            renderer.frame_times.clear()
            reset()

            def compute_frame(index):
                with tf.device(device):
                    reset_started = time.perf_counter()
                    if index == 0:
                        reset()
                    infer_started = time.perf_counter()
                    output = step(*device_frames[index])
                    if gpu_body:
                        cloth, matrices = output[0].numpy()[0, 0], output[1].numpy()
                        body = None
                    else:
                        body, cloth = output[0].numpy()[0, 0], output[1].numpy()[0, 0]
                        matrices = None
                    infer_finished = time.perf_counter()
                    return (body, cloth, matrices, infer_started, infer_finished,
                            (infer_started - reset_started) * 1000)

            if report['pipeline']:
                # Compare the worker against the serial path, including a clip
                # boundary. No OpenGL calls occur in the worker.
                check_indices = list(range(min(count, 32))) + [0, 1 % count]
                expected = [compute_frame(i)[:3] for i in check_indices]
                pipeline = FramePrefetch(compute_frame)
                max_pipeline_error = 0.0
                for i, reference in enumerate(expected):
                    actual = pipeline.take(check_indices[i])[:3]
                    if i + 1 < len(check_indices):
                        pipeline.prefetch(check_indices[i + 1])
                    for a, b in zip(actual, reference):
                        if a is not None:
                            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
                            max_pipeline_error = max(max_pipeline_error, float(np.max(np.abs(a - b))))
                report['validation']['pipeline_max_abs_error'] = max_pipeline_error
                reset()
            index = 0
            next_due = time.perf_counter()
            last_frame = None
            started = time.perf_counter()
            last_title = started
            last_memory = started
            report['memory_samples'] = [memory_sample(tf, device, 0)]
            report['completed'] = False
            report['total_frames'] = 0
            while not renderer.glfw.window_should_close(renderer.window):
                event_started = time.perf_counter()
                renderer.glfw.poll_events()
                event_done = time.perf_counter()
                if args.duration is not None and time.perf_counter() - started >= args.duration:
                    report['completed'] = True
                    report['stop_reason'] = 'duration'
                    break
                if renderer.restart:
                    if pipeline:
                        pipeline.discard()
                    else:
                        reset()
                    index = 0
                    next_due = time.perf_counter()
                    renderer.restart = False
                    renderer.single_step = renderer.paused
                if args.mode == 'benchmark' and args.iterations and report['total_frames'] >= args.iterations:
                    report['completed'] = True
                    break
                now = time.perf_counter()
                ready = renderer.single_step or (not renderer.paused and
                    (args.mode == 'benchmark' or now >= next_due))
                if not ready:
                    if renderer.paused:
                        renderer.draw()
                        renderer.swap()
                        renderer.glfw.wait_events_timeout(0.005)
                    else:
                        # No interpolated display frames: wait for the next model
                        # tick instead of overshooting it with another buffer swap.
                        renderer.glfw.wait_events_timeout(min(max(next_due - now, 0.0001), 0.01))
                    continue
                frame_started = time.perf_counter()
                input_done = time.perf_counter()
                result = pipeline.take(index) if pipeline else compute_frame(index)
                body, cloth, matrices, infer_started, infer_finished, reset_ms = result
                inference_done = time.perf_counter()
                if not (np.isfinite(matrices if gpu_body else body).all() and np.isfinite(cloth).all()):
                    raise RuntimeError('Non-finite predicted vertices')
                bn = vertex_normals(body, model.body.faces) if args.normals == 'cpu' else None
                cn = vertex_normals(cloth, model.garment.faces) if args.normals == 'cpu' else None
                normals_done = time.perf_counter()
                renderer.upload(body, cloth, bn, cn, matrices=matrices)
                upload_done = time.perf_counter()
                normal_query = renderer.update_normals(measure=True)
                normal_submit_done = time.perf_counter()
                query = renderer.draw(measure=True)
                draw_done = time.perf_counter()
                # Let inference run while the native swap call releases the
                # GIL. Starting during Python GL submission caused contention.
                if pipeline and not (args.iterations and report['total_frames'] + 1 >= args.iterations):
                    pipeline.prefetch((index + 1) % count)
                prefetch_done = time.perf_counter()
                renderer.swap()
                finished = time.perf_counter()
                row = {'sample': report['total_frames'], 'clip_frame': index,
                       'width': renderer.ctx.viewport[2], 'height': renderer.ctx.viewport[3],
                       'event_poll_ms': (event_done - event_started) * 1000,
                       'query_collect_ms': 0.0,
                       'input_upload_ms': (input_done - frame_started) * 1000,
                       'inference_readback_ms': (infer_finished - infer_started) * 1000,
                       'inference_wait_ms': (inference_done - input_done) * 1000,
                       'clip_reset_ms': reset_ms,
                       'prefetch_age_ms': max(0, (inference_done - infer_finished) * 1000),
                       'normals_ms': (normals_done - inference_done) * 1000,
                       'vertex_upload_cpu_ms': (upload_done - normals_done) * 1000,
                       'mesh_update_submit_cpu_ms': (normal_submit_done - upload_done) * 1000,
                       'draw_submit_cpu_ms': (draw_done - normal_submit_done) * 1000,
                       'prefetch_submit_ms': (prefetch_done - draw_done) * 1000,
                       'swap_ms': (finished - prefetch_done) * 1000,
                       'processing_ms': (finished - frame_started) * 1000,
                       'inference_to_swap_ms': (finished - infer_started) * 1000,
                       'update_interval_ms': None if last_frame is None else (finished - last_frame) * 1000,
                       'gpu_draw_ms': None, 'gpu_mesh_update_ms': None}
                samples.append(row)
                report['total_frames'] += 1
                if query is not None:
                    pending.append((row, query, normal_query))
                collect_started = time.perf_counter()
                if len(pending) > 4:
                    old, old_query, old_normal_query = pending.popleft()
                    old['gpu_draw_ms'] = old_query.elapsed / 1e6
                    if old_normal_query is not None:
                        old['gpu_mesh_update_ms'] = old_normal_query.elapsed / 1e6
                row['query_collect_ms'] = (time.perf_counter() - collect_started) * 1000
                last_frame = finished
                index = (index + 1) % count
                renderer.single_step = False
                next_due = max(frame_started + config.time_step, time.perf_counter())
                if finished - last_title >= 0.5:
                    recent = list(samples)[-30:]
                    intervals = [x['update_interval_ms'] for x in recent if x['update_interval_ms']]
                    hz = 1000 / np.mean(intervals) if intervals else 0
                    renderer.title(f'NCS | frame {row["clip_frame"]}/{count} | {hz:.1f} updates/s | '
                                   f'{row["processing_ms"]:.2f} ms | {args.mode}')
                    last_title = finished
                if finished - last_memory >= 30:
                    report['memory_samples'].append(memory_sample(tf, device, finished - started))
                    last_memory = finished
            report['session_wall_seconds'] = time.perf_counter() - started
            report['memory_samples'].append(memory_sample(tf, device, report['session_wall_seconds']))
        finally:
            try:
                if pipeline:
                    pipeline.close()
            finally:
                if renderer:
                    try:
                        while pending:
                            row, query, normal_query = pending.popleft()
                            row['gpu_draw_ms'] = query.elapsed / 1e6
                            if normal_query is not None:
                                row['gpu_mesh_update_ms'] = normal_query.elapsed / 1e6
                        save_report(args, report, samples, np)
                    finally:
                        renderer.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
