from keras import backend as K
from keras import initializers
from keras.engine import Layer
from keras.utils import conv_utils


def differentiable_clip(inputs, alpha, rmin, rmax):
    return K.sigmoid(-alpha * (inputs - rmin)) + K.sigmoid(alpha * (inputs - rmax))


def double_thresholding(ghd_layer, inputs, double_threshold, per_pixel=False):
    input_shape = inputs.shape.as_list()

    if double_threshold:
        if per_pixel:
            r = ghd_layer.add_weight(name='r',
                                     shape=input_shape[1:],
                                     dtype=K.floatx(),
                                     initializer=initializers.glorot_normal(807),
                                     regularizer=None,
                                     trainable=True)
        else:
            r = ghd_layer.add_weight(name='r',
                                     shape=(input_shape[-1],),
                                     dtype=K.floatx(),
                                     initializer=initializers.glorot_normal(829),
                                     regularizer=None,
                                     trainable=True)
    else:
        r = ghd_layer.add_weight(name='r',
                                 shape=(input_shape[-1],),
                                 dtype=K.floatx(),
                                 initializer=initializers.zeros(),
                                 regularizer=None,
                                 trainable=False)

    if len(input_shape) == 4:
        axis = (1, 2)
    else:
        axis = (1,)

    rmin = K.min(inputs, axis=axis, keepdims=True) * r
    rmax = K.max(inputs, axis=axis, keepdims=True) * r

    alpha = 0.2

    hout = 0.5 + (inputs - 0.5) * differentiable_clip(inputs, alpha, rmin, rmax)

    if not double_threshold:
        hout = K.relu(0.5 + hout)

    return hout


class ConvGHD(Layer):
    def __init__(self, filters, kernel_size, padding='VALID', double_threshold=True, **kwargs):
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
        hout = double_thresholding(self, hout, self.double_threshold)
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
    def __init__(self, units, double_threshold=True, **kwargs):
        """

        :param filters: number of filters for convolution
        :param kernel_size: weight size in [h,w]
        :param padding: 'SAME' or 'VALID'
        :param kwargs:
        """
        self.units = units
        self.double_threshold = double_threshold
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
        mean_w = K.mean(self.weight, axis=0, keepdims=True)
        mean_x = K.mean(inputs, axis=1, keepdims=True)
        hout = (2. / l) * K.dot(inputs, self.weight) - mean_w - mean_x
        hout = double_thresholding(self, hout, self.double_threshold)
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
