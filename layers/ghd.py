import tensorflow as tf
from keras.engine.topology import Layer
from keras import initializers
from keras.layers import Conv2D, AveragePooling2D, Dense


class DenseGHD(Layer):
    def __init__(self, units, **kwargs):
        self.units = units

        super(DenseGHD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(self.units, use_bias=False)
        super(DenseGHD, self).build(input_shape)

    def call(self, inputs, **kwargs):
        dense_out = self.dense(inputs)
        W = self.dense.kernel
        self.trainable_weights = self.dense.trainable_weights
        self.non_trainable_weights = self.dense.non_trainable_weights
        l = tf.cast(tf.reduce_prod(W.shape), tf.float32)
        hout = (2. / l) * dense_out - tf.reduce_mean(W) - tf.reduce_mean(dense_out, axis=-1, keep_dims=True)
        hout = tf.nn.relu(0.5 + hout)
        return hout

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)


class Conv2DGHD(Layer):
    def __init__(self, filters, kernel_size, strides, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        super(Conv2DGHD, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters),
        #                               initializer='glorot_uniform',
        #                               trainable=True)
        # print self.filters
        # print self.kernel_size
        # print self.strides
        self.conv2d = Conv2D(filters=self.filters,
                             kernel_size=self.kernel_size,
                             strides=self.strides,
                             use_bias=False,
                             input_shape=input_shape)
        self.avgpool2d = AveragePooling2D(pool_size=self.kernel_size, strides=self.strides,
                                          input_shape=input_shape)
        super(Conv2DGHD, self).build(input_shape)

    def call(self, inputs, **kwargs):
        conv2d_out = self.conv2d(inputs, **kwargs)
        avgpool2d_out = tf.reduce_mean(self.avgpool2d(inputs, **kwargs), axis=-1, keep_dims=True)
        W = self.conv2d.kernel
        self.trainable_weights = self.conv2d.trainable_weights
        self.non_trainable_weights = self.conv2d.non_trainable_weights
        l = tf.cast(tf.reduce_prod(W.shape), tf.float32)
        hout = (2. / l) * conv2d_out - tf.reduce_mean(W) - avgpool2d_out
        # hout = tf.nn.relu(0.5 - hout)
        return hout


        # steps_i = (input_shape[1] - self.kernel_size[0]) / self.strides[0] + 1
        # steps_j = (input_shape[2] - self.kernel_size[1]) / self.strides[1] + 1

        # outs = []
        # for f in xrange(self.filters):
        #     for i in xrange(0, steps_i, self.strides[0]):
        #         for j in xrange(0, steps_j, self.strides[1]):
        #             slice_x = inputs[:, i:i + self.kernel_size[0], j:j + self.kernel_size[1]]
        #             x = tf.reshape(slice_x,
        #                            shape=[-1, self.kernel_size[0] * self.kernel_size[1]])
        #             w = tf.reshape(W[:, :, :, f],
        #                            shape=[-1, 1])
        #             out = 2. * tf.matmul(x, w) / l  # [batch, 1]
        #
        #             avg_x = tf.reduce_mean(x, axis=(0,))
        #             avg_w = tf.reduce_mean(w, axis=(0,))
        #
        #             out = out - avg_w - avg_x
        #
        #             outs.append(out)
        # row = tf.concat(row, axis=1)
        # layer.append(row)
        # stack row

        # outs = tf.concat(outs, axis=1)
        # outs = tf.reshape(outs, shape=[-1, steps_i, steps_j, self.filters])
        # stack 2d
        #
        # h = tf.reduce_sum(2. * w * x / l - tf.reduce_mean(w) - tf.reduce_mean(x))

        # return tf.nn.relu((0.5 + h))

    def compute_output_shape(self, input_shape):
        return self.conv2d.compute_output_shape(input_shape)


class GHD(Layer):
    def __init__(self, use_activation=False, per_layer=False, **kwargs):
        self.use_activation = use_activation
        self.per_layer = per_layer

        super(GHD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='weights',
                                      shape=input_shape[1:],
                                      initializer='glorot_uniform',
                                      trainable=True)

        shape = (1,) if self.per_layer else input_shape[1:]

        # if self.use_activation:
        #     self.r = self.add_weight(name='r',
        #                              shape=shape,
        #                              initializer='glorot_uniform',
        #                              trainable=True)

        super(GHD, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input_shape = tf.cast(tf.shape(inputs), tf.float32)
        x = inputs
        w = self.kernel
        l = tf.cast(tf.reduce_prod(w.shape), tf.float32)

        # print w.shape, tf.reduce_mean(w, axis=(0, 1))

        h = 2. * w * x / l - tf.reduce_mean(w) - tf.reduce_mean(x)
        # if self.use_activation:
        #     axis = (0, ) if len(inputs.shape) == 2 else (0,1,2)
        #     o = tf.reduce_max(inputs) if self.per_layer else tf.reduce_max(inputs, axis=axis)
        #     return tf.nn.relu((0.5 + self.r * o) - h)
        # else:
        return tf.nn.relu((0.5 + h))

        # return h

    def compute_output_shape(self, input_shape):
        return input_shape
