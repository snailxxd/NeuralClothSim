"""Geometry checks that run without a display or TensorFlow."""

import unittest
from unittest.mock import PropertyMock, patch
import os
from threading import Event
from types import SimpleNamespace

import numpy as np

from realtime_renderer import Renderer, camera_matrix, vertex_normals, normal_adjacency, is_software_renderer
from view_realtime import FramePrefetch, playback_steps, sample_interpolated_motion
from viewer_ui import PerformanceHistory


class PerformanceHistoryTests(unittest.TestCase):
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

    def test_due_ticks_do_not_depend_on_draw_frequency(self):
        for draw_hz in (10, 60, 180):
            deadline = 0.0
            ticks = 0
            for frame in range(draw_hz + 1):
                steps = playback_steps(frame / draw_hz, deadline, 1 / 30)
                ticks += steps
                deadline += steps / 30
            self.assertEqual(ticks, 31)  # Initial pose plus 30 elapsed ticks.

    def test_late_draw_preserves_all_inference_ticks(self):
        self.assertEqual(playback_steps(0.1, 0, 1 / 30), 4)
        self.assertEqual(playback_steps(0.01, 1 / 30, 1 / 30), 0)


class PrefetchTests(unittest.TestCase):
    def test_serial_order_and_single_frame_bound(self):
        history = []
        def compute(index):
            history.append(index)
            return index * 2
        runner = FramePrefetch(compute)
        try:
            runner.prefetch(0)
            with self.assertRaises(RuntimeError):
                runner.prefetch(1)
            with self.assertRaises(RuntimeError):
                runner.take(1)
            self.assertEqual(runner.take(0), 0)
            runner.prefetch(1)
            self.assertEqual(runner.take(1), 2)
            runner.prefetch(2)
            runner.discard()
            self.assertEqual(runner.take(0), 0)
            self.assertEqual(history, [0, 1, 2, 0])
        finally:
            runner.close()

    def test_discard_drains_running_inference_before_reset(self):
        entered, release = Event(), Event()
        history = []
        def compute(index):
            entered.set()
            if not release.wait(timeout=2):
                raise RuntimeError('Test timed out')
            history.append(index)
        runner = FramePrefetch(compute)
        try:
            runner.prefetch(1)
            self.assertTrue(entered.wait(timeout=2))
            release.set()
            runner.discard()
            history.append('reset')
            self.assertEqual(history, [1, 'reset'])
        finally:
            release.set()
            runner.close()

    def test_worker_error_propagates(self):
        def compute(index):
            raise ValueError('inference failed')
        runner = FramePrefetch(compute)
        try:
            with self.assertRaisesRegex(ValueError, 'inference failed'):
                runner.take(0)
        finally:
            runner.close()


class GeometryTests(unittest.TestCase):
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

    def test_area_weighted_normals_and_isolated_vertex(self):
        vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0],
                             [0, 0, 1], [3, 3, 3]], dtype='f4')
        faces = np.array([[0, 1, 2], [0, 3, 1]], dtype='i4')
        normals = vertex_normals(vertices, faces)
        np.testing.assert_allclose(normals[0], [0, 2**-0.5, 2**-0.5], atol=1e-6)
        np.testing.assert_allclose(normals[2], [0, 0, 1])
        np.testing.assert_allclose(normals[3], [0, 1, 0])
        np.testing.assert_array_equal(normals[4], [0, 0, 0])
        self.assertEqual(normals.dtype, np.float32)

    def test_degenerate_triangles_are_finite(self):
        normals = vertex_normals(np.zeros((3, 3), dtype='f4'), np.array([[0, 1, 2]]))
        np.testing.assert_array_equal(normals, np.zeros((3, 3)))

    def test_camera_projects_center_and_respects_aspect(self):
        center = np.array([0.5, 1, -2], dtype='f4')
        for yaw, pitch in ((0, 0), (1.2, -0.7), (-0.5, 1.4)):
            matrix = camera_matrix(center, 5, yaw, pitch, 16 / 9)
            clip = matrix @ np.append(center, 1)
            self.assertGreater(clip[3], 0)
            np.testing.assert_allclose(clip[:2] / clip[3], [0, 0], atol=1e-6)
            self.assertTrue(-1 < clip[2] / clip[3] < 1)


@unittest.skipUnless(os.environ.get('NCS_TEST_TENSORFLOW') == '1', 'Optional TensorFlow feature test')
class PreviewStateTests(unittest.TestCase):
    def test_many_previews_do_not_advance_recurrent_state(self):
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
                                garment=SimpleNamespace(vertices=0.0), call_spatial_inputs=spatial,
                                call_inputs=lambda poses, trans: (poses[:, -1:], matrices),
                                lbs_cloth=lambda value, matrix: value)
        stream = StreamingNCS(model)
        stream.network_step = lambda features, hidden: (features, hidden + 1)
        def pose(value):
            return tf.constant([[[[value, 0.0, 0.0]]]], dtype=tf.float32)
        trans = tf.zeros([1, 1, 3])
        stream.reset(pose(0.04), trans, pose(0.0), trans)
        stream.hidden.assign([[2]])
        state = [stream.previous_x, stream.previous_j, stream.older_j, stream.hidden]
        before = [v.numpy().copy() for v in state]
        history = tf.concat([pose(0.0), pose(0.01), pose(0.04)], axis=1)
        for _ in range(180):
            _, _, hidden = stream.preview_for_render(history, tf.zeros([1, 3, 3]))
            np.testing.assert_array_equal(hidden.numpy(), [[3]])
        for variable, expected in zip(state, before):
            np.testing.assert_array_equal(variable.numpy(), expected)


@unittest.skipUnless(os.environ.get('NCS_TEST_OPENGL') == '1', 'Requires hardware OpenGL window')
class OpenGLTests(unittest.TestCase):
    def test_gpu_normals_and_imgui(self):
        vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0],
                             [0, 0, 1], [3, 3, 3]], dtype='f4')
        faces = np.array([[0, 1, 2], [0, 3, 1], [0, 0, 0]], dtype='u4')
        renderer = Renderer(vertices, vertices, faces, faces, 800, 700, False)
        try:
            renderer.upload(vertices, vertices)
            query = renderer.update_normals(measure=True)
            for buffer, _, _ in renderer.meshes:
                actual = np.frombuffer(buffer.read(), dtype='f4').reshape(-1, 6)[:, 3:]
                np.testing.assert_allclose(actual, vertex_normals(vertices, faces), atol=1e-6)
            self.assertGreaterEqual(query.elapsed, 0)
            renderer.set_fps(123.4)
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
            np.testing.assert_array_equal(pixels[:, 450:], baseline[:, 450:])
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
            renderer.ui.state['mode'] = 'interpolated'
            with patch.object(renderer.ui.imgui, 'button', side_effect=lambda label: label == 'Pause'), \
                    patch.object(renderer.ui.imgui, 'slider_float', return_value=(True, 2.0)):
                renderer.begin_ui()
                self.assertTrue(renderer.paused)
                self.assertEqual(renderer.playback_speed, 2.0)
                renderer.draw()
        finally:
            renderer.close()


if __name__ == '__main__':
    unittest.main()
