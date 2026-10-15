"""Geometry checks that run without a display or TensorFlow."""

import unittest
from unittest.mock import Mock, PropertyMock, patch
import os
from types import SimpleNamespace

import numpy as np

from .realtime_renderer import Renderer, camera_matrix, normal_adjacency, is_software_renderer
from .cli import save_report, validate_weight_restore
from .playback import PlaybackClock, FixedStepPlayback, DisplaySkeleton, sample_interpolated_motion
from .viewer_ui import PerformanceHistory


def reference_normals(vertices, faces):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    indices = faces.ravel()
    values = np.repeat(normals, 3, axis=0)
    result = np.column_stack([np.bincount(indices, weights=values[:, axis],
                                          minlength=len(vertices)) for axis in range(3)])
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result.astype('f4')


class PerformanceHistoryTests(unittest.TestCase):
    def test_weight_restore_accepts_hdf5_and_validates_tensorflow_status(self):
        validate_weight_restore(None)
        status = SimpleNamespace(
            assert_existing_objects_matched=Mock(),
            expect_partial=Mock(),
        )
        validate_weight_restore(status)
        status.assert_existing_objects_matched.assert_called_once_with()
        status.expect_partial.assert_called_once_with()

    def test_report_counts_real_model_steps_and_flattens_timings(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(json=Path(directory) / 'report.json', csv=Path(directory) / 'report.csv')
            rows = [{'processing_ms': 5, 'inference_readback_ms': 3,
                     'inference_samples_ms': [2, 4]},
                    {'processing_ms': 5, 'inference_readback_ms': None,
                     'inference_samples_ms': []}]
            report = {'model_fps': 30, 'session_wall_seconds': 1, 'total_frames': 2,
                      'simulation_steps': 2}
            with contextlib.redirect_stdout(io.StringIO()):
                save_report(args, report, rows, np)
            saved = json.loads(args.json.read_text())
            self.assertEqual(saved['summary']['inference_readback_ms']['mean'], 3)
            self.assertAlmostEqual(saved['summary']['inference_readback_ms']['p95'], 3.9)
            self.assertEqual(saved['summary']['observed_model_steps_per_second'], 2)
            self.assertTrue(args.csv.exists())

    def test_delayed_queries_and_rolling_statistics(self):
        history = PerformanceHistory()
        row = {'processing_ms': 2, 'gpu_draw_ms': None}
        history.append(row, now=0)
        history.append({'processing_ms': 4}, now=0.5)
        stats = history.snapshot(now=0.5)
        self.assertEqual(stats['processing_ms'][0], 3)
        self.assertAlmostEqual(stats['processing_ms'][1], 3.9)
        self.assertEqual(stats['inference_hz'], 2)
        self.assertNotIn('gpu_draw_ms', stats)
        row['gpu_draw_ms'] = 1.5
        self.assertEqual(history.snapshot(now=0.8)['gpu_draw_ms'], (1.5, 1.5))
        for i in range(300):
            history.append({}, now=i + 1)
        self.assertEqual(len(history.samples), 240)

    def test_idle_draws_do_not_dilute_model_time(self):
        history = PerformanceHistory()
        history.append({'inference_readback_ms': 3, 'inference_samples_ms': [2, 4],
                        'steps_advanced': 2}, now=0)
        history.append({'inference_readback_ms': None, 'inference_samples_ms': [],
                        'steps_advanced': 0}, now=0.1)
        self.assertEqual(history.snapshot(now=0.2)['inference_readback_ms'][0], 3)
        self.assertEqual(history.snapshot(now=0.2)['inference_hz'], 0)


class PlaybackTests(unittest.TestCase):
    def test_input_slerp_and_translation_midpoint(self):
        sequence = SimpleNamespace(fps=30, duration=1/30, num_frames=2,
                                   poses=np.array([[[1,0,0,0]], [[0,0,1,0]]]),
                                   trans=np.array([[0,0,0], [2,0,0]]))
        pose, trans = sample_interpolated_motion(sequence, 1/60)
        np.testing.assert_allclose(pose[0,0], [2**-0.5,0,2**-0.5,0], atol=1e-6)
        np.testing.assert_allclose(trans[0], [1,0,0])
        self.assertEqual(pose.dtype, np.float32)

    def test_antipodal_quaternions_do_not_take_long_arc(self):
        sequence = SimpleNamespace(fps=30, duration=1/30, num_frames=2,
                                   poses=np.array([[[1,0,0,0]], [[-1,0,0,0]]]),
                                   trans=np.zeros((2,3)))
        pose, _ = sample_interpolated_motion(sequence, 1/60)
        np.testing.assert_allclose(pose[0,0], [1,0,0,0])

    def test_ticks_and_interpolation_independent_of_render_rate(self):
        for draw_hz in (10, 60, 180):
            calls = []
            def compute(index):
                calls.append(index)
                return np.array([index], dtype='f4')
            playback = FixedStepPlayback(100, 1 / 30, compute)
            for i in range(draw_hz + 1):
                before, after, alpha, source, _ = playback.sample(i / draw_hz)
                np.testing.assert_allclose((1-alpha)*before + alpha*after, [source*30], atol=1e-5)
            self.assertEqual(calls, list(range(30)))

    def test_loop_reset_single_frame_and_stall(self):
        calls = []
        playback = FixedStepPlayback(3, 0.1, lambda i: calls.append(i) or i)
        playback.sample(0)
        before, after, alpha, _, _ = playback.sample(0.25)
        self.assertEqual((before, after), (1, 2))
        self.assertAlmostEqual(alpha, 0.5)
        before, after, alpha, _, computed = playback.sample(0.41)
        self.assertEqual((before, after), (0, 1))
        self.assertAlmostEqual(alpha, 0.1)
        self.assertEqual(calls, [0, 1, 2, 0, 1])
        single = FixedStepPlayback(1, 0.1, lambda i: i)
        self.assertEqual(single.sample(0.15)[:3], (0, 0, 0.0))
        playback = FixedStepPlayback(100, 0.1, lambda i: i)
        self.assertEqual(playback.sample(0.65)[4], list(range(7)))

    def test_pause_resume_speed_and_step_clock(self):
        clock = PlaybackClock(0)
        self.assertEqual(clock.tick(1, False, 1), 1)
        self.assertEqual(clock.tick(2, True, 1), 1)
        self.assertEqual(clock.tick(100, True, 1, 0.1), 1.1)
        self.assertEqual(clock.tick(101, False, 1), 1.1)
        self.assertAlmostEqual(clock.tick(102, False, 2), 3.1)


class CameraControlTests(unittest.TestCase):
    def setUp(self):
        self.buttons, self.keys = {}, {}
        self.renderer = Renderer.__new__(Renderer)
        self.renderer.glfw = SimpleNamespace(
            PRESS=1, MOUSE_BUTTON_LEFT=0, MOUSE_BUTTON_RIGHT=1, MOUSE_BUTTON_MIDDLE=2,
            KEY_LEFT_SHIFT=340, KEY_RIGHT_SHIFT=344,
            get_mouse_button=lambda window, button: self.buttons.get(button, 0),
            get_key=lambda window, key: self.keys.get(key, 0),
            get_window_size=lambda window: (800, 600))
        self.renderer.window = None
        self.renderer.ui = None
        self.renderer.center = np.array([0, 1, 0], dtype='f4')
        self.renderer.radius = 5
        self.renderer.yaw, self.renderer.pitch = 0.0, 0.15
        self.renderer.cursor = (100, 100)
        self.renderer.initial_camera = (self.renderer.center.copy(), 5, 0.0, 0.15)

    def test_pan_follows_screen_drag_at_different_camera_angles(self):
        renderer = self.renderer
        for yaw, pitch in ((0, 0), (1.2, 0.7), (-2, -1.1)):
            renderer.yaw, renderer.pitch = yaw, pitch
            point = np.append(renderer.center.copy(), 1)
            renderer.pan_camera(80, 60)
            clip = camera_matrix(renderer.center, 5, yaw, pitch, 800 / 600) @ point
            np.testing.assert_allclose(clip[:2] / clip[3], [0.2, -0.2], atol=1e-6)

    def test_drag_modes_and_ui_capture(self):
        renderer = self.renderer
        for button, shift in ((2, False), (1, False), (0, True)):
            renderer.reset_camera()
            renderer.cursor = (100, 100)
            self.buttons.clear()
            self.keys.clear()
            self.buttons[button] = 1
            if shift:
                self.keys[340] = 1
            renderer._cursor(None, 120, 130)
            self.assertFalse(np.array_equal(renderer.center, renderer.initial_camera[0]))
            self.assertEqual((renderer.yaw, renderer.pitch), (0.0, 0.15))
        renderer.ui = SimpleNamespace(captures_mouse=True)
        center = renderer.center.copy()
        renderer._cursor(None, 150, 160)
        np.testing.assert_array_equal(renderer.center, center)
        renderer.ui = None
        self.keys.clear()
        renderer._cursor(None, 170, 180)
        self.assertNotEqual(renderer.yaw, 0)
        np.testing.assert_array_equal(renderer.center, center)

    def test_reset_restores_initial_view(self):
        renderer = self.renderer
        renderer.pan_camera(100, 100)
        renderer.radius, renderer.yaw, renderer.pitch = 10, 1, 1
        renderer.reset_camera()
        np.testing.assert_array_equal(renderer.center, renderer.initial_camera[0])
        self.assertEqual((renderer.radius, renderer.yaw, renderer.pitch), (5, 0, 0.15))
        self.assertIsNone(renderer.cursor)


class GeometryTests(unittest.TestCase):
    def test_display_fk_preserves_bone_length_during_rotation(self):
        local = np.tile(np.eye(4, dtype='f4'), (2, 1, 1))
        local[1, 0, 3] = 1
        body = SimpleNamespace(rotations_local=local, rotations_local_inv=np.linalg.inv(local),
                               parents=np.array([-1, 0]), depth=[[1]])
        sequence = SimpleNamespace(fps=1, duration=1, num_frames=2,
                                   poses=np.array([[[1,0,0,0], [1,0,0,0]],
                                                   [[0,0,0,1], [1,0,0,0]]]),
                                   trans=np.zeros((2,3)))
        pose, trans = sample_interpolated_motion(sequence, 0.5)
        matrices = DisplaySkeleton(body).matrices(pose, trans)
        child = matrices[1] @ np.array([1, 0, 0, 1])
        np.testing.assert_allclose(child, [0, 1, 0], atol=1e-6)
        self.assertAlmostEqual(np.linalg.norm(child), 1, places=6)

    def test_adjacency_covers_shared_and_isolated_vertices(self):
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype='u4')
        offsets, adjacent = normal_adjacency(faces, 5)
        expected = [[0, 1], [0], [0, 1], [1], []]
        for i, values in enumerate(expected):
            self.assertEqual(list(adjacent[offsets[i]:offsets[i+1]]), values)

    def test_software_detection(self):
        self.assertTrue(is_software_renderer('llvmpipe (LLVM 21.1)'))
        self.assertTrue(is_software_renderer('Microsoft Basic Render Driver'))
        self.assertFalse(is_software_renderer('D3D12 (NVIDIA GeForce RTX 3060 Ti)'))

    def test_camera_projects_center_and_respects_aspect(self):
        center = np.array([0.5, 1, -2], dtype='f4')
        for yaw, pitch in ((0, 0), (1.2, -0.7), (-0.5, 1.4)):
            matrix = camera_matrix(center, 5, yaw, pitch, 16 / 9)
            clip = matrix @ np.append(center, 1)
            self.assertGreater(clip[3], 0)
            np.testing.assert_allclose(clip[:2] / clip[3], [0, 0], atol=1e-6)
            self.assertTrue(-1 < clip[2] / clip[3] < 1)


@unittest.skipUnless(os.environ.get('NCS_TEST_TENSORFLOW') == '1', 'Optional TensorFlow feature test')
class LocalOutputTests(unittest.TestCase):
    def test_motion_updates_traced_network_without_changing_recurrence(self):
        import sys
        from pathlib import Path
        import tensorflow as tf
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ncs'))
        from model.streaming import StreamingNCS
        gru = tf.keras.layers.GRU(3, use_bias=False)
        gru.build((1, 1, 12))
        gru.set_weights([np.full(weight.shape, 0.1, dtype='f4') for weight in gru.get_weights()])
        model = SimpleNamespace(body=SimpleNamespace(num_input_joints=1),
                                static_encoder=[lambda x: tf.reshape(x[..., :3], [1, 1, 3])],
                                dynamic_encoder=[lambda x: tf.reshape(x, [1, 1, 12]), gru],
                                decoder=[])
        stream = StreamingNCS(model)
        features = tf.ones([1, 1, 1, 21])
        hidden = tf.zeros([1, 3])
        for compiled in (False, True):
            network = (tf.function(stream.network_step.python_function, jit_compile=True)
                       if compiled else stream.network_step)
            stream.motion.assign(1.0)
            baseline, state = network(features, hidden)
            traces = network.experimental_get_tracing_count()
            for motion in (0.0, 0.5, 2.0, 1.0):
                stream.motion.assign(motion)
                output, current_state = network(features, hidden)
                np.testing.assert_allclose(output.numpy(), 1 + motion * (baseline.numpy() - 1),
                                           rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal(current_state.numpy(), state.numpy())
                self.assertEqual(network.experimental_get_tracing_count(), traces)
            self.assertGreater(float(tf.norm(state)), 0)

    def test_local_api_preserves_fixed_step_state_and_world_output(self):
        import sys
        from pathlib import Path
        import tensorflow as tf
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ncs'))
        from model.streaming import StreamingNCS
        eye = tf.eye(3, batch_shape=[1, 1, 1])
        matrices = tf.concat([eye, tf.zeros([1, 1, 1, 3, 1])], axis=-1)
        def spatial(poses, trans):
            return tf.concat([poses] * 3, axis=-1), poses, matrices, eye
        model = SimpleNamespace(body=SimpleNamespace(num_input_joints=1, input_joints=[0]),
                                dynamic_encoder=[SimpleNamespace(units=1)],
                                config=SimpleNamespace(time_step=0.2),
                                garment=SimpleNamespace(vertices=1.0), call_spatial_inputs=spatial,
                                lbs_cloth=lambda value, matrix: value * 2)
        stream = StreamingNCS(model)
        stream.network_step = lambda features, hidden: (features[..., :3], hidden + 1)
        def pose(value):
            return tf.constant([[[[value, 0.0, 0.0]]]], dtype=tf.float32)
        trans = tf.zeros([1, 1, 3])
        stream.reset(pose(0), trans, pose(0), trans)
        local, bones, hidden = stream.step_local(pose(0.04), trans)
        saved = [v.numpy().copy() for v in (stream.previous_x, stream.previous_j, stream.older_j, stream.hidden)]
        stream.reset(pose(0), trans, pose(0), trans)
        world, expected_hidden = stream.step(pose(0.04), trans)
        np.testing.assert_allclose(world.numpy(), local.numpy() * 2)
        np.testing.assert_array_equal(hidden.numpy(), expected_hidden.numpy())
        for variable, expected in zip((stream.previous_x, stream.previous_j, stream.older_j, stream.hidden), saved):
            np.testing.assert_array_equal(variable.numpy(), expected)


@unittest.skipUnless(os.environ.get('NCS_TEST_OPENGL') == '1', 'Requires hardware OpenGL window')
class OpenGLTests(unittest.TestCase):
    def test_gpu_local_shape_interpolation_before_skinning(self):
        vertices = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype='f4')
        faces = np.array([[0,1,2]], dtype='u4')
        body_weights = np.tile([1, 0], (3, 1)).astype('f4')
        cloth_weights = np.tile([0, 1], (3, 1)).astype('f4')
        renderer = Renderer(vertices, vertices, faces, faces, 640, 480, False,
                            body_rest=vertices, body_weights=body_weights,
                            cloth_weights=cloth_weights, ui=False)
        try:
            matrices = np.array([[[1,0,0,0], [0,1,0,0], [0,0,1,0]],
                                 [[0,-1,0,2], [1,0,0,0], [0,0,1,0]]], dtype='f4')
            following = vertices + [0.5, 0.25, 0]
            renderer.upload_local_pair(vertices, following)
            for alpha in (0, 0.5, 1):
                renderer.upload_display(matrices, alpha)
                renderer.update_mesh()
                local = (1-alpha)*vertices + alpha*following
                expected = local @ matrices[1, :, :3].T + matrices[1, :, 3]
                actual = np.frombuffer(renderer.meshes[1][0].read(), dtype='f4').reshape(-1, 6)[:, :3]
                np.testing.assert_allclose(actual, expected, atol=1e-6)
                body = np.frombuffer(renderer.meshes[0][0].read(), dtype='f4').reshape(-1, 6)[:, :3]
                np.testing.assert_allclose(body, vertices, atol=1e-6)
        finally:
            renderer.close()

    def test_gpu_normals_and_imgui(self):
        vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0],
                             [0, 0, 1], [3, 3, 3]], dtype='f4')
        faces = np.array([[0, 1, 2], [0, 3, 1], [0, 0, 0]], dtype='u4')
        weights = np.ones((len(vertices), 1), dtype='f4')
        renderer = Renderer(vertices, vertices, faces, faces, 800, 700, False,
                            body_rest=vertices, body_weights=weights, cloth_weights=weights)
        try:
            query = renderer.update_mesh(measure=True)
            for buffer, _, _ in renderer.meshes:
                actual = np.frombuffer(buffer.read(), dtype='f4').reshape(-1, 6)[:, 3:]
                np.testing.assert_allclose(actual, reference_normals(vertices, faces), atol=1e-6)
            self.assertGreaterEqual(query.elapsed, 0)
            renderer.fps = 123.4
            from OpenGL import GL
            panel = renderer.ui
            # Run consecutive frames to check state restoration across renderers.
            for _ in range(3):
                renderer.glfw.poll_events()
                renderer.begin_ui()
                renderer.draw()
                self.assertEqual(renderer.ctx.error, 'GL_NO_ERROR')
                renderer.swap()
            renderer.draw()
            width, height = renderer.glfw.get_framebuffer_size(renderer.window)
            pixels = np.frombuffer(GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE),
                                   dtype='u1').reshape(height, width, 3)[::-1].copy()
            renderer.ui = None
            renderer.draw()
            baseline = np.frombuffer(GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE),
                                     dtype='u1').reshape(height, width, 3)[::-1].copy()
            renderer.ui = panel
            np.testing.assert_array_equal(pixels[:, 480:], baseline[:, 480:])
            self.assertGreater(np.count_nonzero(pixels[35:110, 20:400, 0] > 200), 100)
            self.assertGreater(renderer.ui.imgui.get_draw_data().total_vtx_count, 100)
            # Panel capture suppresses camera zoom; the same scroll updates ImGui.
            io = renderer.ui.imgui.get_io()
            radius = renderer.radius
            with patch.object(type(renderer.ui), 'captures_mouse', new_callable=PropertyMock,
                              return_value=True):
                renderer._scroll(renderer.window, 0, 1)
            self.assertEqual(renderer.radius, radius)
            self.assertEqual(io.mouse_wheel, 1)
            renderer.swap()
            with patch.object(renderer.ui.imgui, 'button', side_effect=lambda label: label == 'Pause'), \
                    patch.object(renderer.ui.imgui, 'slider_float', return_value=(True, 2.0)):
                renderer.begin_ui()
                self.assertTrue(renderer.paused)
                self.assertEqual(renderer.playback_speed, 2.0)
                self.assertEqual(renderer.motion, 2.0)
                renderer.draw()
            renderer.swap()
            with patch.object(renderer.ui.imgui, 'button',
                              side_effect=lambda label: label == 'Reset motion (1.0)'):
                renderer.begin_ui()
                self.assertEqual(renderer.motion, 1.0)
                renderer.draw()
        finally:
            renderer.close()


if __name__ == '__main__':
    unittest.main()
