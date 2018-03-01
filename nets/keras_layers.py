from keras import backend as K
from keras import initializers
from keras.engine import Layer
from keras.utils import conv_utils


def differentiable_clip(inputs, alpha, rmin, rmax):
    """
    Output from this clipping function is something like below:
    ___            ___
       \          /
        \        /
         \      /
          \____/
    """
    return K.sigmoid(-alpha * (inputs - rmin)) + K.sigmoid(alpha * (inputs - rmax))


def double_thresholding(ghd_layer, inputs):
    input_shape = inputs.shape.as_list()

    shape = input_shape[1:] if ghd_layer.per_pixel else (input_shape[-1],)

    rmin = ghd_layer.add_weight(name='rmin',
                                shape=shape,
                                dtype=K.floatx(),
                                initializer=initializers.glorot_normal(807),
                                regularizer=None,
                                trainable=ghd_layer.double_threshold)
    rmax = ghd_layer.add_weight(name='rmax',
                                shape=shape,
                                dtype=K.floatx(),
                                initializer=initializers.glorot_normal(807),
                                regularizer=None,
                                trainable=ghd_layer.double_threshold)

    if len(input_shape) == 4:
        axis = (1, 2)
    else:
        axis = (1,)

    inputs_rmin = K.min(inputs, axis=axis, keepdims=True) * K.sigmoid(rmin)
    inputs_rmax = K.max(inputs, axis=axis, keepdims=True) * K.sigmoid(rmax)

    alpha = ghd_layer.alpha

    hout = 0.5 + (inputs - 0.5) * differentiable_clip(inputs, alpha, inputs_rmin, inputs_rmax)

    return hout


class CustomRelu(Layer):
    def __init__(self, threshold=0.5, sign=1, **kwargs):
        self.threshold = threshold
        self.sign = sign
        super(CustomRelu, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        out = K.relu(self.threshold + inputs * self.sign)
        return out


class ConvGHD(Layer):
    def __init__(self, filters, kernel_size, padding='VALID', double_threshold=True, per_pixel=False, alpha=0.2,
                 **kwargs):
        """

        :param filters: number of filters for convolution
        :param kernel_size: weight size in [h,w]
        :param padding: 'SAME' or 'VALID'
        :param kwargs:
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding.lower()
        self.double_threshold = double_threshold
        self.per_pixel = per_pixel
        self.alpha = alpha
        super(ConvGHD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(name='conv_ghd_weight',
                                      shape=[self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters],
                                      dtype=K.floatx(),
                                      initializer=initializers.glorot_normal(4567),
                                      regularizer=None,
                                      trainable=True)

        self.mean_weight = self.add_weight(name='mean_conv_ghd_weight',
                                           shape=[self.kernel_size[0], self.kernel_size[1], input_shape[-1], 1],
                                           dtype=K.floatx(),
                                           initializer=initializers.constant(1.0),
                                           regularizer=None,
                                           trainable=False)

        super(ConvGHD, self).build(input_shape)

    def call(self, inputs, **kwargs):
        conv = K.conv2d(inputs,
                        kernel=self.weight,
                        strides=[1, 1],
                        padding=self.padding)

        # if weight shape is [5,5,1,16], l will be 5*5*1
        l = K.constant(reduce(lambda x, y: x * y,
                              self.weight.shape.as_list()[:3],
                              1),
                       dtype=K.floatx())

        # convolution way of mean
        # output shape will be (batch, height, width, 1)
        # it can be achieved by conv2d with constant weights
        mean_x = 1. / l * K.conv2d(inputs,
                                   kernel=self.mean_weight,
                                   strides=[1, 1],
                                   padding=self.padding)

        # mean for every filter, output shape will be (16,)
        mean_w = K.mean(self.weight, axis=(0, 1, 2), keepdims=True)
        hout = (2. / l) * conv - mean_w - mean_x
        hout = double_thresholding(self, hout)
        return hout

    def compute_output_shape(self, input_shape):
        """
        compute output shape copied from Conv2D
        :param input_shape:
        :return:
        """
        space = input_shape[1:-1]
        new_space = []
        strides = [1, 1]
        dilation_rate = [1, 1]
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=strides[i],
                dilation=dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)


class FCGHD(Layer):
    def __init__(self, units, double_threshold=True, per_pixel=False, alpha=0.2, **kwargs):
        """

        :param filters: number of filters for convolution
        :param kernel_size: weight size in [h,w]
        :param padding: 'SAME' or 'VALID'
        :param kwargs:
        """
        self.units = units
        self.double_threshold = double_threshold
        self.per_pixel = per_pixel
        self.alpha = alpha
        super(FCGHD, self).__init__(**kwargs)

    def build(self, input_shape):
        self._input_shape = input_shape
        self.weight = self.add_weight(name='fc_ghd_weight',
                                      shape=[input_shape[1], self.units],
                                      dtype=K.floatx(),
                                      initializer=initializers.glorot_normal(1234),
                                      regularizer=None,
                                      trainable=True)

        super(FCGHD, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # fc version of ghd is easier than convolution
        input_shape = self._input_shape
        l = K.constant(input_shape[1],
                       dtype=K.floatx())
        mean_x = K.mean(inputs, axis=1, keepdims=True)
        mean_w = K.mean(self.weight, axis=0, keepdims=True)
        hout = (2. / l) * K.dot(inputs, self.weight) - mean_w - mean_x
        hout = double_thresholding(self, hout)
        return hout

    def compute_output_shape(self, input_shape):
        """
        compute output shape copied from Dense
        :param input_shape:
        :return:
        """
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
