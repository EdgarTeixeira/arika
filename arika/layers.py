from tensorflow import custom_gradient, identity
from tensorflow.keras.layers import Layer


@custom_gradient
def gradient_reversal(x):
    """Helper function for the GradientReversal layer"""

    def grad(dy):
        return -dy

    return identity(x), grad


class GradientReversal(Layer):
    """
    A keras layer that during the foward pass applies the identity function
    and during da backward pass inverts the gradient (-grad)
    """

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        return gradient_reversal(x)

    def compute_output_shape(self, input_shape):
        return input_shape
