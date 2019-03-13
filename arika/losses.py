import tensorflow as tf

_EPSILON = 1e-7


def binary_focal_loss(gamma=2.0):
    def _focal_loss(y_true, y_pred):
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        correction = tf.pow(1 - p_t, gamma)
        p_t = tf.clip_by_value(p_t, _EPSILON, 1.0 - _EPSILON)
        return -tf.reduce_mean(correction * tf.log(p_t))

    return _focal_loss


def categorical_focal_loss(gamma=2.0):
    def _focal_loss(y_true, y_pred):
        outs = tf.clip_by_value(y_pred, _EPSILON, 1.0 - _EPSILON)
        correction = tf.pow(1 - outs, gamma)
        outs = y_true * tf.log(outs)
        return -tf.reduce_sum(correction * outs, axis=-1)

    return _focal_loss
