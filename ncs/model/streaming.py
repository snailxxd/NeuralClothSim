"""Stateful inference with cached spatial features; one instance per character."""

import tensorflow as tf


class StreamingNCS(tf.Module):
    """Wrap a built/restored NCS model without changing its checkpoint weights.

    Inputs to reset/step have shapes (B, 1, J, D) and (B, 1, 3).
    reset receives two historical frames; step receives the next current frame.
    At a clip's start, repeat its first pose as both historical frames, matching
    Data(mode='test'). Reset again on teleports or independent clip playback.
    Cache and GRU state stay on the device. Calls on one instance must be serial.
    """

    def __init__(self, model, motion=1.0, batch_size=1):
        super().__init__()
        self.model = model
        self.motion = motion
        joints = model.body.num_input_joints
        self.previous_x = tf.Variable(tf.zeros([batch_size, 1, joints, 9]), trainable=False)
        self.previous_j = tf.Variable(tf.zeros([batch_size, 1, joints, 3]), trainable=False)
        self.older_j = tf.Variable(tf.zeros([batch_size, 1, joints, 3]), trainable=False)
        self.hidden = tf.Variable(tf.zeros([batch_size, model.dynamic_encoder[-1].units]),
                                  trainable=False)
        self.initialized = tf.Variable(False, trainable=False)

    def spatial(self, poses, trans):
        x, j, matrices, inverse = self.model.call_spatial_inputs(poses, trans)
        indices = self.model.body.input_joints
        return tuple(tf.gather(value, indices, axis=axis) for value, axis in
                     ((x, -2), (j, -2), (matrices, -3), (inverse, -3)))

    @tf.function(reduce_retracing=True)
    def reset(self, older_pose, older_trans, previous_pose, previous_trans):
        _, older_j, _, _ = self.spatial(older_pose, older_trans)
        previous_x, previous_j, _, _ = self.spatial(previous_pose, previous_trans)
        self.previous_x.assign(previous_x)
        self.previous_j.assign(previous_j)
        self.older_j.assign(older_j)
        self.hidden.assign(tf.zeros_like(self.hidden))
        self.initialized.assign(True)
        return self.initialized.read_value()

    @tf.function(reduce_retracing=True)
    def network_step(self, features, hidden):
        static, dynamic = tf.split(features, [9, 12], axis=-1)
        for layer in self.model.static_encoder:
            static = layer(static)
        for layer in self.model.dynamic_encoder[:-1]:
            dynamic = layer(dynamic)
        dynamic, states = self.model.dynamic_encoder[-1].cell(
            dynamic[:, 0], [hidden], training=False)
        value = static + self.motion * dynamic[:, None]
        for layer in self.model.decoder:
            value = layer(value)
        return value, states[0]

    @tf.function(reduce_retracing=True)
    def step(self, poses, trans):
        """Return cloth and hidden, preserving the original streaming API."""
        return self._step(poses, trans, include_body=False)

    @tf.function(reduce_retracing=True)
    def step_with_body(self, poses, trans):
        """Return (body, cloth, hidden), reusing the full-skeleton FK matrices."""
        return self._step(poses, trans, include_body=True)

    @tf.function(reduce_retracing=True)
    def step_for_render(self, poses, trans):
        """Return (cloth, full bone matrices, hidden) for renderer-side body LBS."""
        return self._step(poses, trans, include_body=False, return_matrices=True)

    @tf.function(reduce_retracing=True)
    def preview_for_render(self, poses, trans):
        """Read-only prediction from three poses spaced by config.time_step.

        Unlike step(), the preview never changes spatial caches or GRU state.
        Commit complete model ticks with step() before previewing fractional time.
        """
        return self._preview(poses, trans, include_body=False)

    @tf.function(reduce_retracing=True)
    def preview_with_body(self, poses, trans):
        return self._preview(poses, trans, include_body=True)

    def _preview(self, poses, trans, include_body):
        tf.debugging.assert_equal(self.initialized, True, message="Call reset() before preview()")
        tf.debugging.assert_equal(tf.shape(poses)[1], 3, message="preview() accepts three historical poses")
        features, full_matrices = self.model.call_inputs(poses, trans)
        network = getattr(self, "preview_network_step", self.network_step)
        deformation, hidden = network(features, self.hidden.read_value())
        matrices = tf.gather(full_matrices, self.model.body.input_joints, axis=-3)
        cloth = self.model.lbs_cloth(self.model.garment.vertices + deformation, matrices)
        if include_body:
            return self.model.lbs_body(self.model.body.vertices, full_matrices), cloth, hidden
        return cloth, full_matrices, hidden

    def _step(self, poses, trans, include_body, return_matrices=False):
        tf.debugging.assert_equal(self.initialized, True, message="Call reset() before step()")
        tf.debugging.assert_equal(tf.shape(poses)[1], 1, message="step() accepts one frame")
        x, j, full_matrices, inverse = self.model.call_spatial_inputs(poses, trans)
        indices = self.model.body.input_joints
        x = tf.gather(x, indices, axis=-2)
        j = tf.gather(j, indices, axis=-2)
        matrices = tf.gather(full_matrices, indices, axis=-3)
        inverse = tf.gather(inverse, indices, axis=-3)
        dt = self.model.config.time_step
        dx = (x - self.previous_x) / dt
        # Match the original repeated-difference order to minimize rounding drift.
        dj = ((j - self.previous_j) / dt - (self.previous_j - self.older_j) / dt) / dt
        dj = (inverse @ dj[..., None])[..., 0]
        features = tf.concat((x, dx, dj), axis=-1)
        deformation, hidden = self.network_step(features, self.hidden.read_value())
        cloth = self.model.lbs_cloth(self.model.garment.vertices + deformation, matrices)
        self.older_j.assign(self.previous_j)
        self.previous_j.assign(j)
        self.previous_x.assign(x)
        self.hidden.assign(hidden)
        if return_matrices:
            return cloth, full_matrices, hidden
        if include_body:
            body = self.model.lbs_body(self.model.body.vertices, full_matrices)
            return body, cloth, hidden
        return cloth, hidden
