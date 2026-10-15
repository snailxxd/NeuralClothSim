"""Dear ImGui controls and rolling performance statistics for the GLFW viewer."""

from collections import deque
import time

import numpy as np


class PerformanceHistory:
    FIELDS = (
        ('processing_ms', 'Frame processing'),
        ('event_poll_ms', 'Events + UI build'),
        ('inference_readback_ms', 'Inference + readback'),
        ('inference_wait_ms', 'Model tick wait'),
        ('display_pose_ms', 'Display SLERP / FK'),
        ('vertex_upload_cpu_ms', 'Upload'),
        ('mesh_update_submit_cpu_ms', 'Mesh submission'),
        ('draw_submit_cpu_ms', 'Draw submission'),
        ('swap_ms', 'Swap'),
        ('gpu_mesh_update_ms', 'GPU skin / normals'),
        ('gpu_draw_ms', 'GPU draw + UI'),
    )

    def __init__(self):
        self.samples = deque(maxlen=240)
        self.updated = -float('inf')
        self.cached = {}

    def append(self, sample, now=None):
        # Keep references: delayed GPU query results fill these rows later.
        self.samples.append((time.perf_counter() if now is None else now, sample))

    def snapshot(self, now=None):
        now = time.perf_counter() if now is None else now
        if now - self.updated < 0.25:
            return self.cached
        self.updated = now
        result = {}
        for key, _ in self.FIELDS:
            values = [value for _, row in self.samples
                      for value in (row.get('inference_samples_ms', [row.get(key)])
                                    if key == 'inference_readback_ms' else [row.get(key)])
                      if value is not None]
            if values:
                result[key] = (float(np.mean(values)), float(np.percentile(values, 95)))
        recent = [(stamp, row) for stamp, row in self.samples if now - stamp <= 1]
        result['inference_hz'] = (sum(row.get('steps_advanced', 1) for _, row in recent[1:]) /
                                  (recent[-1][0] - recent[0][0])
                                  if len(recent) > 1 and recent[-1][0] > recent[0][0] else 0.0)
        self.cached = result
        return result


class ViewerUI:
    def __init__(self, renderer):
        import os
        import sys
        # WSL may advertise Wayland while GLFW uses X11. Match the active
        # context instead of PyOpenGL's environment-based platform guess.
        if sys.platform.startswith('linux'):
            glfw = renderer.glfw
            platform = glfw.get_platform()
            if platform == glfw.PLATFORM_X11:
                os.environ.setdefault('PYOPENGL_PLATFORM', 'glx')
            elif platform == glfw.PLATFORM_WAYLAND:
                os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        import imgui
        from imgui.integrations.glfw import GlfwRenderer
        self.imgui = imgui
        self.owner = renderer
        self.context = imgui.create_context()
        imgui.get_io().ini_file_name = None
        imgui.style_colors_dark()
        try:
            self.backend = GlfwRenderer(renderer.window, attach_callbacks=False)
        except Exception:
            imgui.destroy_context(self.context)
            raise
        self.prepared = False
        self.history = PerformanceHistory()
        self.state = {}

    @property
    def captures_mouse(self):
        return self.imgui.get_io().want_capture_mouse

    @property
    def captures_keyboard(self):
        return self.imgui.get_io().want_capture_keyboard

    def prepare(self):
        if self.prepared:
            return
        ui, owner = self.imgui, self.owner
        self.backend.process_inputs()
        ui.new_frame()
        self.prepared = True
        ui.set_next_window_position(12, 12, condition=ui.FIRST_USE_EVER)
        ui.set_next_window_size(450, 660, condition=ui.FIRST_USE_EVER)
        ui.begin('NeuralClothSim | Performance')
        fps = owner.fps or 0.0
        ui.text(f'FPS: {fps:.1f}    Frame: {1000 / fps if fps else 0:.2f} ms')
        stats = self.history.snapshot()
        ui.text(f'Inference: {0 if owner.paused else stats.get("inference_hz", 0):.1f} / s')
        ui.separator()
        if ui.button('Resume' if owner.paused else 'Pause'):
            owner.paused = not owner.paused
        ui.same_line()
        if ui.button('Restart'):
            owner.restart = True
        ui.same_line()
        if ui.button('Step'):
            owner.paused, owner.single_step = True, True
        changed, speed = ui.slider_float('Playback speed', owner.playback_speed, 0.1, 4.0, '%.2fx')
        if changed:
            owner.playback_speed = speed
        if ui.button('Reset speed (1x)'):
            owner.playback_speed = 1.0
        changed, motion = ui.slider_float('Motion', owner.motion, 0.0, 2.0, '%.2f')
        if changed:
            owner.motion = motion
        if ui.button('Reset motion (1.0)'):
            owner.motion = 1.0
        ui.text_disabled('Dynamic cloth strength; applies on the next model tick.')
        if ui.button('Reset camera (F)'):
            owner.reset_camera()
        ui.text(f'Animation: {self.state.get("animation_time", 0):.2f} s')
        ui.text(f'Model step: {self.state.get("model_fps", 0):g} Hz (fixed)')
        ui.text(f'Display delay: {self.state.get("display_delay_ms", 0):.1f} animation ms')
        inference = stats.get('inference_readback_ms')
        if inference and inference[0] > 0:
            budget = 1000 / max(self.state.get('model_fps', 30), 1e-6)
            ui.text(f'Model budget used: {100 * inference[0] / budget:.1f}%')
            ui.text(f'Serial step estimate: {1000 / inference[0]:.0f} / s')
        changed, wireframe = ui.checkbox('Wireframe', owner.wireframe)
        if changed:
            owner.wireframe = wireframe
        ui.same_line()
        changed, vsync = ui.checkbox('VSync', owner.vsync)
        if changed:
            owner.vsync = vsync
            owner.glfw.swap_interval(int(vsync))
        ui.separator()
        if len(owner.frame_times) > 1:
            intervals = (np.diff(np.asarray(owner.frame_times)) * 1000).astype('f4')
            ui.plot_lines('Frame ms', intervals, scale_min=0,
                          scale_max=max(25, float(np.percentile(intervals, 95)) * 1.2),
                          graph_size=(330, 55))
        ui.columns(3, 'timings', border=False)
        ui.set_column_width(0, 220)
        ui.set_column_width(1, 85)
        for text in ('Stage', 'Mean ms', 'P95 ms'):
            ui.text(text)
            ui.next_column()
        ui.separator()
        for key, label in PerformanceHistory.FIELDS:
            values = stats.get(key)
            ui.text(label)
            ui.next_column()
            ui.text(f'{values[0]:.3f}' if values else '--')
            ui.next_column()
            ui.text(f'{values[1]:.3f}' if values else '--')
            ui.next_column()
        ui.columns(1)
        ui.text_disabled('Last 240 draws; inference excludes idle draws.')
        ui.text_disabled('Serial estimate excludes rendering / scheduling.')
        ui.separator()
        ui.text_wrapped(owner.ctx.info.get('GL_RENDERER', 'Unknown GPU'))
        ui.text_disabled('Left drag: orbit | Wheel: zoom')
        ui.text_disabled('Middle / right / Shift+left drag: pan | F: reset view')
        ui.end()

    def render(self):
        self.prepare()
        self.imgui.render()
        self.backend.render(self.imgui.get_draw_data())
        self.prepared = False

    def close(self):
        if self.prepared:
            self.imgui.end_frame()
        self.backend.shutdown()
        self.imgui.destroy_context(self.context)
