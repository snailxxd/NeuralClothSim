"""Fixed model ticks and smooth display poses, independent of draw frequency."""

import math
import numpy as np


def sample_interpolated_motion(sequence, seconds):
    """Shortest-arc quaternion SLERP and linear root translation at wall time."""
    frame = float(np.clip(seconds, 0, sequence.duration)) * float(sequence.fps)
    index = min(int(frame), sequence.num_frames - 1)
    following = min(index + 1, sequence.num_frames - 1)
    alpha = frame - index
    q0 = np.asarray(sequence.poses[index], dtype=np.float64)
    q1 = np.asarray(sequence.poses[following], dtype=np.float64)
    q0 = q0 / np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-12)
    q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-12)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    denominator = np.maximum(np.sin(angle), 1e-12)
    spherical = (np.sin((1 - alpha) * angle) * q0 + np.sin(alpha * angle) * q1) / denominator
    pose = np.where(dot > 0.9995, (1 - alpha) * q0 + alpha * q1, spherical)
    pose /= np.maximum(np.linalg.norm(pose, axis=-1, keepdims=True), 1e-12)
    trans = (1 - alpha) * sequence.trans[index] + alpha * sequence.trans[following]
    return pose.astype(np.float32)[None], np.asarray(trans, dtype=np.float32)[None]


class PlaybackClock:
    def __init__(self, now):
        self.time = 0.0
        self.last = now
        self.paused = False

    def tick(self, now, paused, speed, step=0.0):
        if not paused and not self.paused:
            self.time += max(0.0, now - self.last) * speed
        self.time += step
        self.last, self.paused = now, paused
        return self.time


class FixedStepPlayback:
    """Keep only two sequential inference results; delay display by one tick.

    compute(index) advances one fixed model tick and resets at index zero.
    Independent clip loops never interpolate across their reset boundary.
    """
    def __init__(self, count, time_step, compute):
        self.count, self.dt, self.compute = count, time_step, compute
        self.loop = None
        self.index = -1
        self.previous = self.current = None

    def sample(self, animation_time):
        display_time = max(0.0, animation_time - self.dt)
        loop = int(display_time / (self.count * self.dt))
        source_time = display_time - loop * self.count * self.dt
        frame = min(self.count - 1, source_time / self.dt)
        upper = min(self.count - 1, int(math.ceil(frame - 1e-9)))
        if loop != self.loop:
            self.loop, self.index = loop, -1
            self.previous = self.current = None
        computed = []
        while self.index < upper:
            self.index += 1
            result = self.compute(self.index)
            self.previous = self.current if self.current is not None else result
            self.current = result
            computed.append(result)
        alpha = float(np.clip(frame - max(0, upper - 1), 0, 1)) if upper else 0.0
        return self.previous, self.current, alpha, source_time, computed


class DisplaySkeleton:
    """NumPy FK for display only; uses the model's rest transforms and hierarchy.

    Quaternion SLERP happens before FK, so interpolated bones retain their
    lengths. No TensorFlow call or GRU update occurs at display frequency.
    """
    def __init__(self, body):
        self.local = np.asarray(body.rotations_local, dtype='f4')
        self.inverse = np.asarray(body.rotations_local_inv, dtype='f4')
        self.parents = body.parents
        self.depth = body.depth

    def matrices(self, pose, translation):
        q = np.asarray(pose, dtype='f4').reshape(-1, 4)
        w, x, y, z = q.T
        tx, ty, tz = 2*x, 2*y, 2*z
        rotations = np.stack((1-(ty*y+tz*z), tx*y-tz*w, tx*z+ty*w,
                              tx*y+tz*w, 1-(tx*x+tz*z), ty*z-tx*w,
                              tx*z-ty*w, ty*z+tx*w, 1-(tx*x+ty*y)), axis=-1)
        basis = np.tile(np.eye(4, dtype='f4'), (len(q), 1, 1))
        basis[:, :3, :3] = rotations.reshape(-1, 3, 3)
        basis[0, :3, 3] = np.asarray(translation).reshape(3)
        global_pose = np.empty_like(basis)
        global_pose[0] = self.local[0] @ basis[0]
        for depth in self.depth:
            parent = self.parents[depth]
            global_pose[depth] = ((global_pose[parent] @ self.inverse[parent])
                                  @ self.local[depth]) @ basis[depth]
        return np.ascontiguousarray((global_pose @ self.inverse)[:, :3])
