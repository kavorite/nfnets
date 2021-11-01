import tensorflow as tf


def serializable(cls):
    return tf.keras.utils.register_keras_serializable(
        package="nfnets", name=cls.__name__
    )(cls)


@serializable
class WSConv2D(tf.keras.layers.Conv2D):
    """WSConv2d
    Reference: https://github.com/deepmind/deepmind-research/blob/master/nfnets/base.py#L121
    """

    def __init__(self, *args, **kwargs):
        super(WSConv2D, self).__init__(
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="untruncated_normal",
            ),
            *args,
            **kwargs
        )
        # Get gain
        self.gain = self.add_weight(
            name="gain",
            shape=(self.filters,),
            initializer="ones",
            trainable=True,
            dtype=self.dtype,
        )

    def standardize_weight(self, eps):
        mean, var = tf.nn.moments(self.kernel, axes=[0, 1, 2], keepdims=True)
        fan_in = tf.cast(tf.reduce_prod(self.kernel.shape[:-1]), self.dtype)

        # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var)
        scale = (
            tf.math.rsqrt(
                tf.math.maximum(
                    var * fan_in, tf.convert_to_tensor(eps, dtype=self.dtype)
                )
            )
            * self.gain
        )
        shift = mean * scale
        return self.kernel * scale - shift

    def call(self, inputs, eps=1e-4):
        weight = self.standardize_weight(eps)
        return (
            tf.nn.conv2d(
                inputs,
                weight,
                strides=self.strides,
                padding=self.padding.upper(),
                dilations=self.dilation_rate,
            )
            + self.bias
        )


@serializable
class SqueezeExcite(tf.keras.layers.Layer):
    """Simple Squeeze+Excite module."""

    def __init__(
        self,
        in_ch,
        out_ch,
        se_ratio=0.5,
        hidden_ch=None,
        activation=tf.nn.relu,
        name=None,
    ):
        super(SqueezeExcite, self).__init__(name=name)
        self.in_ch, self.out_ch = in_ch, out_ch
        if se_ratio is None:
            if hidden_ch is None:
                raise ValueError("Must provide one of se_ratio or hidden_ch")
            self.hidden_ch = hidden_ch
        else:
            self.hidden_ch = max(1, int(self.in_ch * se_ratio))
        self.activation = activation
        self.fc0 = tf.keras.layers.Dense(self.hidden_ch, use_bias=True)
        self.fc1 = tf.keras.layers.Dense(self.out_ch, use_bias=True)

    def get_config(self):
        return {
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "activation": self.activation,
        }

    def call(self, x):
        h = tf.math.reduce_mean(x, axis=[1, 2])  # Mean pool over HW extent
        h = self.fc1(self.activation(self.fc0(h)))
        h = tf.keras.activations.sigmoid(h)[:, None, None]  # Broadcast along H, W
        return h


@serializable
class StochDepth(tf.keras.layers.Layer):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    def __init__(self, drop_rate, scale_by_keep=False, name=None):
        super(StochDepth, self).__init__(name=name)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def get_config(self):
        return {
            "drop_rate": self.drop_rate,
            "scale_by_keep": self.scale_by_keep,
        }

    def call(self, x, training):
        if not training:
            return x
        batch_size = tf.shape(x)[0]
        r = tf.random.uniform(shape=[batch_size, 1, 1, 1], dtype=x.dtype)
        keep_prob = 1.0 - self.drop_rate
        binary_tensor = tf.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor
