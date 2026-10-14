"""Small OpenGL viewer, independent of TensorFlow and model loading."""

import math
import time
from collections import deque
import numpy as np


def camera_matrix(center, radius, yaw, pitch, aspect):
    direction = np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch),
                          math.cos(pitch) * math.cos(yaw)], dtype='f4')
    eye = center + radius * direction
    right = np.cross([0, 1, 0], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    view = np.eye(4, dtype='f4')
    view[:3, :3] = np.stack([right, up, direction])
    view[:3, 3] = -view[:3, :3] @ eye
    near, far = max(radius * 0.001, 0.001), max(radius * 100, 100)
    scale = 1 / math.tan(math.radians(45) / 2)
    projection = np.zeros((4, 4), dtype='f4')
    projection[0, 0], projection[1, 1] = scale / aspect, scale
    projection[2, 2] = -(far + near) / (far - near)
    projection[2, 3] = -2 * far * near / (far - near)
    projection[3, 2] = -1
    return projection @ view


VERTEX_SHADER = '''
#version 330
uniform mat4 mvp;
in vec3 position;
in vec3 normal;
out vec3 surface_normal;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    surface_normal = normal;
}
'''
FRAGMENT_SHADER = '''
#version 330
uniform vec3 color;
in vec3 surface_normal;
out vec4 frag_color;
void main() {
    vec3 n = normalize(surface_normal);
    if (!gl_FrontFacing) n = -n;
    float diffuse = max(dot(n, normalize(vec3(0.4, 1.0, 0.7))), 0.0);
    frag_color = vec4(color * (0.3 + 0.7 * diffuse), 1.0);
}
'''

NORMAL_SHADER = '''
#version 430
layout(local_size_x=128) in;
layout(std430, binding=0) buffer Mesh { float data[]; };
layout(std430, binding=1) readonly buffer Faces { uint faces[]; };
layout(std430, binding=2) readonly buffer Offsets { uint offsets[]; };
layout(std430, binding=3) readonly buffer Adjacency { uint adjacent[]; };
uniform uint vertex_count;
vec3 position(uint i) {
    return vec3(data[6*i], data[6*i+1], data[6*i+2]);
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= vertex_count) return;
    vec3 n = vec3(0);
    for (uint j=offsets[i]; j<offsets[i+1]; ++j) {
        uint f = 3 * adjacent[j];
        vec3 a = position(faces[f]);
        n += cross(position(faces[f+1])-a, position(faces[f+2])-a);
    }
    n /= max(length(n), 1e-12);
    data[6*i+3]=n.x; data[6*i+4]=n.y; data[6*i+5]=n.z;
}
'''

SKIN_SHADER = '''
#version 430
layout(local_size_x=128) in;
layout(std430, binding=0) buffer Mesh { float data[]; };
layout(std430, binding=1) readonly buffer Rest { float rest[]; };
layout(std430, binding=2) readonly buffer Weights { float weights[]; };
layout(std430, binding=3) readonly buffer Matrices { float matrices[]; };
layout(std430, binding=4) readonly buffer NextRest { float next_rest[]; };
uniform uint vertex_count;
uniform uint joint_count;
uniform float alpha;
void main() {
    uint i=gl_GlobalInvocationID.x;
    if (i>=vertex_count) return;
    vec3 p=vec3(rest[3*i],rest[3*i+1],rest[3*i+2]);
    p=mix(p,vec3(next_rest[3*i],next_rest[3*i+1],next_rest[3*i+2]),alpha);
    // Match TensorFlow LBS: blend matrices first, then transform the vertex.
    float m[12];
    for (uint k=0; k<12; ++k) m[k]=0;
    for (uint j=0; j<joint_count; ++j) {
        float w=weights[i*joint_count+j];
        for (uint k=0; k<12; ++k) m[k]+=w*matrices[j*12+k];
    }
    for (uint k=0; k<3; ++k)
        data[6*i+k]=m[k*4]*p.x+m[k*4+1]*p.y+m[k*4+2]*p.z+m[k*4+3];
}
'''

def normal_adjacency(faces, vertex_count):
    indices = np.asarray(faces, dtype='u4').ravel()
    order = np.argsort(indices, kind='stable')
    offsets = np.concatenate(([0], np.cumsum(np.bincount(indices, minlength=vertex_count))))
    return offsets.astype('u4'), (order // 3).astype('u4')


def is_software_renderer(name):
    return any(token in name.lower() for token in
               ('llvmpipe', 'softpipe', 'software', 'swrast', 'swiftshader', 'basic render'))


class GLFWLoader:
    """Resolve GL through the active GLFW context, including Linux runtime-only installs."""

    def __init__(self, glfw):
        self.load_opengl_function = glfw.get_proc_address

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def release(self):
        # The window owns the GL context and is destroyed by Renderer.close().
        pass


class Renderer:
    def __init__(self, body, cloth, body_faces, cloth_faces, width, height, vsync, *,
                 body_rest, body_weights, cloth_weights, require_hardware=True,
                 ui=True, playback_speed=1.0):
        import glfw
        import moderngl
        self.glfw, self.gl = glfw, moderngl
        self.window = None
        self.ctx = None
        self.resources = []
        self.ui = None
        self.fps = None
        self.playback_speed = playback_speed
        self.vsync = vsync
        self.normal_buffers = []
        self.frame_times = deque(maxlen=120)
        self.last_fps_update = 0
        if not glfw.init():
            raise RuntimeError('GLFW initialization failed. Check DISPLAY/Wayland and GPU drivers.')
        try:
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            self.window = glfw.create_window(width, height, 'NCS realtime viewer', None, None)
            if not self.window:
                raise RuntimeError('Could not create the required OpenGL window (GPU normals require 4.3).')
            glfw.make_context_current(self.window)
            glfw.swap_interval(int(vsync))
            self.ctx = moderngl.create_context(require=430,
                                              context=GLFWLoader(glfw))
            name = self.ctx.info.get('GL_RENDERER', 'unknown')
            if require_hardware and is_software_renderer(name):
                raise RuntimeError(f'Hardware OpenGL required, got {name}. Check WSL D3D12/driver '
                                   'configuration; --render-device software allows diagnostic fallback.')
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.program = self.ctx.program(vertex_shader=VERTEX_SHADER,
                                            fragment_shader=FRAGMENT_SHADER)
            self.resources.append(self.program)
            self.normal_program = self.ctx.compute_shader(NORMAL_SHADER)
            self.resources.append(self.normal_program)
            self.meshes = []
            for vertices, faces, color in ((body, body_faces, (0.65, 0.65, 0.65)),
                                            (cloth, cloth_faces, (0.25, 0.45, 0.95))):
                # STATIC_DRAW is a placement hint, not immutable storage.
                # On tested Mesa D3D12 this hint is much cheaper than
                # DYNAMIC_DRAW for uploads and compute accesses.
                buffer = self.ctx.buffer(reserve=len(vertices) * 6 * 4)
                indices = self.ctx.buffer(np.asarray(faces, dtype='u4').tobytes())
                vao = self.ctx.vertex_array(self.program,
                    [(buffer, '3f 3f', 'position', 'normal')], indices)
                self.resources.extend([buffer, indices, vao])
                self.meshes.append((buffer, vao, color))
                offsets, adjacent = normal_adjacency(faces, len(vertices))
                offsets_buffer = self.ctx.buffer(offsets.tobytes())
                adjacent_buffer = self.ctx.buffer(adjacent.tobytes())
                self.resources.extend([offsets_buffer, adjacent_buffer])
                self.normal_buffers.append((indices, offsets_buffer, adjacent_buffer, len(vertices)))
            self.skin_program = self.ctx.compute_shader(SKIN_SHADER)
            rest = self.ctx.buffer(np.asarray(body_rest, dtype='f4').tobytes())
            weights = self.ctx.buffer(np.asarray(body_weights, dtype='f4').tobytes())
            matrices = self.ctx.buffer(reserve=body_weights.shape[1] * 12 * 4)
            self.skin_buffers = (rest, weights, matrices)
            self.skin_program['joint_count'].value = body_weights.shape[1]
            previous = self.ctx.buffer(reserve=len(cloth) * 3 * 4)
            following = self.ctx.buffer(reserve=len(cloth) * 3 * 4)
            cloth_weights_buffer = self.ctx.buffer(np.asarray(cloth_weights, dtype='f4').tobytes())
            self.cloth_skin_buffers = (previous, cloth_weights_buffer, matrices, following)
            self.resources.extend([self.skin_program, rest, weights, matrices,
                                   previous, following, cloth_weights_buffer])
            lo, hi = body.min(axis=0), body.max(axis=0)
            self.center = (lo + hi) / 2
            self.radius = max(float(np.linalg.norm(hi - lo)) * 1.6, 0.1)
            size = max(float(np.linalg.norm(hi - lo)), 0.1)
            lines = []
            for offset in np.linspace(-2 * size, 2 * size, 21):
                lines.extend([[self.center[0] + offset, lo[1], self.center[2] - 2 * size],
                              [self.center[0] + offset, lo[1], self.center[2] + 2 * size],
                              [self.center[0] - 2 * size, lo[1], self.center[2] + offset],
                              [self.center[0] + 2 * size, lo[1], self.center[2] + offset]])
            grid = np.column_stack([np.asarray(lines, dtype='f4'),
                                    np.tile([0, 1, 0], (len(lines), 1))]).astype('f4')
            grid_buffer = self.ctx.buffer(grid.tobytes())
            self.grid = self.ctx.vertex_array(self.program,
                [(grid_buffer, '3f 3f', 'position', 'normal')])
            self.resources.extend([grid_buffer, self.grid])
            self.yaw, self.pitch = 0.0, 0.15
            self.paused = self.single_step = self.restart = self.wireframe = False
            self.cursor = None
            if ui:
                from viewer_ui import ViewerUI
                self.ui = ViewerUI(self)
                glfw.set_char_callback(self.window, self.ui.backend.char_callback)
            glfw.set_key_callback(self.window, self._key)
            glfw.set_cursor_pos_callback(self.window, self._cursor)
            glfw.set_scroll_callback(self.window, self._scroll)
            self.upload_local_pair(cloth, cloth)
            identity = np.tile(np.eye(4, dtype='f4')[:3], (body_weights.shape[1], 1, 1))
            self.upload_display(identity, 0)
            self.update_mesh()
        except Exception:
            self.close()
            raise

    def _key(self, window, key, scancode, action, modifiers):
        g = self.glfw
        if self.ui and 0 <= key < 512:
            self.ui.backend.keyboard_callback(window, key, scancode, action, modifiers)
        if action != g.PRESS:
            return
        if key == g.KEY_ESCAPE:
            g.set_window_should_close(window, True)
        elif self.ui and self.ui.captures_keyboard:
            return
        elif key == g.KEY_SPACE:
            self.paused = not self.paused
        elif key == g.KEY_RIGHT:
            self.paused, self.single_step = True, True
        elif key == g.KEY_R:
            self.restart = True
        elif key == g.KEY_W:
            self.wireframe = not self.wireframe

    def _cursor(self, window, x, y):
        if (self.cursor and not (self.ui and self.ui.captures_mouse)
                and self.glfw.get_mouse_button(window, self.glfw.MOUSE_BUTTON_LEFT)):
            self.yaw -= (x - self.cursor[0]) * 0.006
            self.pitch = float(np.clip(self.pitch + (y - self.cursor[1]) * 0.006, -1.4, 1.4))
        self.cursor = (x, y)

    def _scroll(self, window, dx, dy):
        if self.ui:
            self.ui.backend.scroll_callback(window, dx, dy)
            if self.ui.captures_mouse:
                return
        self.radius = float(np.clip(self.radius * math.exp(-dy * 0.1), 0.02, 1e4))

    def update_mesh(self, measure=False):
        """Interpolate local cloth, skin both meshes, then generate normals."""
        query = self.ctx.query(time=True) if measure else None
        if query:
            query.__enter__()
        try:
            skin_inputs = ((*self.skin_buffers, self.skin_buffers[0]), self.cloth_skin_buffers)
            for index, (mesh, buffers) in enumerate(zip(self.meshes, skin_inputs)):
                for binding, buffer in enumerate((mesh[0], *buffers)):
                    buffer.bind_to_storage_buffer(binding)
                count = self.normal_buffers[index][3]
                self.skin_program['vertex_count'].value = count
                self.skin_program['alpha'].value = self.interpolation_alpha if index else 0.0
                self.skin_program.run(group_x=(count + 127) // 128)
            self.ctx.memory_barrier(self.gl.SHADER_STORAGE_BARRIER_BIT)
            for mesh, (faces, offsets, adjacent, count) in zip(self.meshes, self.normal_buffers):
                for binding, buffer in enumerate((mesh[0], faces, offsets, adjacent)):
                    buffer.bind_to_storage_buffer(binding)
                self.normal_program['vertex_count'].value = count
                self.normal_program.run(group_x=(count + 127) // 128)
            self.ctx.memory_barrier(self.gl.VERTEX_ATTRIB_ARRAY_BARRIER_BIT |
                                    self.gl.BUFFER_UPDATE_BARRIER_BIT)
        finally:
            if query:
                query.__exit__(None, None, None)
        return query

    def begin_ui(self):
        if self.ui:
            self.ui.prepare()

    def draw(self, measure=False):
        width, height = self.glfw.get_framebuffer_size(self.window)
        if width < 1 or height < 1:
            return None
        self.ctx.viewport = (0, 0, width, height)
        mvp = camera_matrix(self.center, self.radius, self.yaw, self.pitch, width / height)
        self.program['mvp'].write(mvp.T.astype('f4').tobytes())
        query = self.ctx.query(time=True) if measure else None
        if query:
            query.__enter__()
        try:
            self.ctx.clear(0.08, 0.09, 0.12)
            self.ctx.wireframe = False
            self.program['color'].value = (0.25, 0.28, 0.30)
            self.grid.render(mode=self.gl.LINES)
            self.ctx.wireframe = self.wireframe
            for _, vao, color in self.meshes:
                self.program['color'].value = color
                vao.render()
            self.ctx.wireframe = False
            if self.ui:
                self.ui.render()
        finally:
            if query:
                query.__exit__(None, None, None)
        return query

    def swap(self):
        self.glfw.swap_buffers(self.window)
        now = time.perf_counter()
        self.frame_times.append(now)
        if now - self.last_fps_update >= 0.25 and len(self.frame_times) > 1:
            self.fps = (len(self.frame_times) - 1) / (now - self.frame_times[0])
            self.last_fps_update = now

    def upload_local_pair(self, previous, following):
        """Only upload cloth endpoints when a fixed model tick advances."""
        for buffer, value in zip((self.cloth_skin_buffers[0], self.cloth_skin_buffers[3]),
                                 (previous, following)):
            buffer.orphan()
            buffer.write(np.asarray(value, dtype='f4').tobytes())

    def upload_display(self, matrices, alpha):
        self.skin_buffers[2].orphan()
        self.skin_buffers[2].write(np.asarray(matrices, dtype='f4').tobytes())
        self.interpolation_alpha = alpha

    def close(self):
        if self.ui:
            self.ui.close()
            self.ui = None
        for resource in reversed(self.resources):
            resource.release()
        self.resources.clear()
        if self.ctx:
            self.ctx.release()
            self.ctx = None
        if self.window:
            self.glfw.destroy_window(self.window)
            self.window = None
        self.glfw.terminate()
