"""Geometry checks that run without a display or TensorFlow."""

import unittest
import os

import numpy as np

from realtime_renderer import Renderer, camera_matrix, vertex_normals, normal_adjacency, is_software_renderer


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


@unittest.skipUnless(os.environ.get('NCS_TEST_OPENGL') == '1', 'Requires hardware OpenGL window')
class OpenGLTests(unittest.TestCase):
    def test_gpu_normals_and_hud(self):
        vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0],
                             [0, 0, 1], [3, 3, 3]], dtype='f4')
        faces = np.array([[0, 1, 2], [0, 3, 1], [0, 0, 0]], dtype='u4')
        renderer = Renderer(vertices, vertices, faces, faces, 640, 360, False)
        try:
            renderer.upload(vertices, vertices)
            query = renderer.update_normals(measure=True)
            for buffer, _, _ in renderer.meshes:
                actual = np.frombuffer(buffer.read(), dtype='f4').reshape(-1, 6)[:, 3:]
                np.testing.assert_allclose(actual, vertex_normals(vertices, faces), atol=1e-6)
            self.assertGreaterEqual(query.elapsed, 0)
            renderer.set_fps(123.4)
            renderer.draw()
            width, height = renderer.glfw.get_framebuffer_size(renderer.window)
            pixels = np.frombuffer(renderer.ctx.screen.read(components=3, alignment=1),
                                   dtype='u1').reshape(height, width, 3)[::-1]
            self.assertGreater(np.count_nonzero(pixels[10:40, 10:210, 0] > 200), 100)
            renderer.swap()
        finally:
            renderer.close()


if __name__ == '__main__':
    unittest.main()
