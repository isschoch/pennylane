import tensorflow
from typing import Optional
from collections.abc import Iterable


class QuantumConvolutionalLayer(tensorflow.keras.layers.Layer):
    """Quantum Convolutional Layer.
    A Keras Tensorflow implementation of a trainable quantum convolutional neural network.
    Args:
        qnode (@qml.qnode qnode): qnode which acts as kernel
        weight_shapes (np.ndarray): Shape of weights used in the qnode function.
        kernel_size (int or tuple): Dimensions of kernel.
        strides (int or tuple): Number of pixels by which kernel is shifted after applying kernel to patch.
        padding ("VALID" or "SAME"): The type of padding algorithm to use.
        dilation_rate (int or tuple): Dilation rate, i.e. number of pixels skipped by kernel, of convolution.
        weight_initializer (see keras.initializers): Initializer for the weights of the qnode. Defaults to "glorot_uniform".
    """

    def __init__(
        self,
        qnode,
        weight_shapes: dict,
        kernel_size,
        strides=(1, 1),
        padding="VALID",
        dilation_rate=(1, 1),
        weight_specs: Optional[dict] = None,
        **kwargs
    ):
        # Inherits the initialization of the keras.Layer class
        super(QuantumConvolutionalLayer, self).__init__()

        # Handle integer to 2-element tuple conversion
        def _integer_tuple_handler(input):
            if isinstance(input, tuple) is False and isinstance(input, list) is False:
                input = (input, input)
            return input

        self.kernel_size = _integer_tuple_handler(kernel_size)
        self.dilation_rate = _integer_tuple_handler(dilation_rate)
        self.padding = padding

        self.strides = strides

        self.weight_shapes = {
            weight: (
                tuple(size)
                if isinstance(size, Iterable)
                else (size,)
                if size > 1
                else ()
            )
            for weight, size in weight_shapes.items()
        }
        self.weight_specs = weight_specs if weight_specs is not None else {}
        self.qnode = qnode
        self.qnode_weights = {}

    def build(self, input_shape):
        for weight, size in self.weight_shapes.items():
            spec = self.weight_specs.get(weight, {})
            self.qnode_weights[weight] = self.add_weight(
                name=weight, shape=size, **spec
            )

        super().build(input_shape)

    def _evaluate_qnode(self, x):
        kwargs = {
            **{self.input_arg: x},
            **{k: 1.0 * w for k, w in self.qnode_weights.items()},
        }
        return self.qnode(**kwargs)

    def draw(self):
        print(self.qnode.qnode.draw())

    def call(self, inputs):
        patches = tensorflow.image.extract_patches(
            images=tensorflow.cast(inputs, tensorflow.float64),
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, self.dilation_rate[0], self.dilation_rate[1], 1],
            padding=self.padding,
        )
        return tensorflow.vectorized_map(
            lambda row: tensorflow.vectorized_map(
                lambda patch: tensorflow.vectorized_map(self._evaluate_qnode, patch),
                row,
            ),
            patches,
        )

    _input_arg = "inputs"

    @property
    def input_arg(self):
        return self._input_arg