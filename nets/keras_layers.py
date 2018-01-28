import numpy as np
from keras import backend as K
from keras import initializers
from keras.engine import Layer
from keras.utils import conv_utils


def double_thresholding(hout, double_threshold):
    pass


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
                                      dtype=np.float32,
                                      initializer=initializers.glorot_normal(4567),
                                      regularizer=None,
                                      trainable=True)

        self.mean_weight = self.add_weight(name='mean_conv_ghd_weight',
                                           shape=[self.kernel_size[0], self.kernel_size[1], input_shape[-1], 1],
                                           dtype=np.float32,
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
                       dtype=np.float32)

        # convolution way of mean, avg pool will produce [h, w, c]
        # and we need mean of a block [5,5,channel], so we take mean of avg pooled image at channel axis
        # output shape will be (batch, height, width, 1)

        # mean_x = tf.reduce_mean(tf.nn.avg_pool(inputs,
        #                                        ksize=[1, kernel_size[0], kernel_size[1], 1],
        #                                        strides=[1, 1, 1, 1],
        #                                        padding='VALID'), axis=-1, keep_dims=True)

        # or it can be achieved by conv2d with constant weights
        mean_x = 1. / l * K.conv2d(inputs,
                                   kernel=self.mean_weight,
                                   strides=[1, 1],
                                   padding=self.padding)

        # mean for every filter, output shape will be (16,)
        mean_w = K.mean(self.weight, axis=(0, 1, 2), keepdims=True)
        hout = (2. / l) * conv - mean_w - mean_x
        hout = double_thresholding(hout, self.double_threshold)
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
        self.weight = self.add_weight(name='fc_ghd_weight',
                                      shape=[input_shape[1], self.units],
                                      dtype=np.float32,
                                      initializer=initializers.glorot_normal(1234),
                                      regularizer=None,
                                      trainable=True)

        super(FCGHD, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # fc version of ghd is easier than convolution
        l = K.constant(inputs.shape.as_list()[1],
                       dtype=np.float32)
        mean_w = K.mean(self.weight, axis=0, keepdims=True)
        mean_x = K.mean(inputs, axis=1, keepdims=True)
        hout = (2. / l) * K.dot(inputs, self.weight) - mean_w - mean_x
        hout = double_thresholding(hout, self.double_threshold)
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
