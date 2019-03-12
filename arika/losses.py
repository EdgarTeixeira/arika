import tensorflow as tf


def focal_loss(gamma=2.0):
    def _focal_loss(y_true, y_pred):
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        correction = tf.pow(1 - p_t, gamma)
        p_t = tf.clip_by_value(p_t, 1e-7, 1.0)
        return tf.reduce_mean(-correction * tf.log(p_t))
    return _focal_loss
