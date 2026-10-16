import tensorflow as tf
from tensorflow.keras.metrics import Metric


class MyMetric(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name="m/" + name, **kwargs)
        self.error = self.add_weight(name="error", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, error, sample_weight=1.0):
        weight = tf.cast(sample_weight, self.dtype)
        self.error.assign_add(tf.cast(error, self.dtype) * weight)
        self.count.assign_add(weight)

    def result(self):
        return tf.math.divide_no_nan(self.error, self.count)

    def load_own_variables(self, store):
        # Older best.weights.h5 files have no counters for newly added metrics.
        # Only log counters may be absent; network weights still load strictly.
        if len(store) == 0:
            self.reset_state()
        else:
            super().load_own_variables(store)
