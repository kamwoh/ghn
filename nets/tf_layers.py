import tensorflow as tf


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(y_pred, y_true)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc


def differentiable_clip(inputs, alpha, rmin, rmax):
    return tf.sigmoid(-alpha * (inputs - rmin)) + tf.sigmoid(alpha * (inputs - rmax))


def double_thresholding(inputs, name, double_threshold=False, per_pixel=True):
    input_shape = inputs.shape.as_list()

    if double_threshold:
        if per_pixel:
            r = tf.get_variable(name=name + '_r',
                                shape=input_shape[1:],
                                dtype=tf.float32,
                                initializer=tf.glorot_normal_initializer(807),
                                regularizer=None,
                                trainable=True)
        else:
            r = tf.get_variable(name=name + '_r',
                                shape=(input_shape[-1],),
                                dtype=tf.float32,
                                initializer=tf.glorot_normal_initializer(829),
                                regularizer=None,
                                trainable=True)
    else:
        r = tf.get_variable(name=name + '_r',
                            shape=(input_shape[-1],),
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer(),
                            regularizer=None,
                            trainable=False)

    if len(input_shape) == 4:
        axis = (1, 2)
    else:
        axis = (1,)

    rmin = tf.reduce_min(inputs, axis=axis, keep_dims=True) * r
    rmax = tf.reduce_max(inputs, axis=axis, keep_dims=True) * r

    alpha = 0.2

    hout = 0.5 + (inputs - 0.5) * differentiable_clip(inputs, alpha, rmin, rmax)
    # hout = tf.nn.relu(hout)
    if not double_threshold:
        hout = tf.nn.relu(0.5 + hout)

    return hout


def conv_ghd(inputs, filters, kernel_size, name, with_ghd=True, with_relu=True, double_threshold=False):
    conv_weight = tf.get_variable(name=name + '_weights',
                                  shape=[kernel_size[0], kernel_size[1], inputs.shape.as_list()[-1], filters],
                                  dtype=tf.float32,
                                  initializer=tf.glorot_normal_initializer(4567),
                                  regularizer=None,
                                  trainable=True)

    conv = tf.nn.conv2d(inputs, conv_weight,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name=name)

    if with_ghd:
        # if weight shape is [5,5,1,16], l will be 5*5*1
        l = tf.constant(reduce(lambda x, y: x * y, conv_weight.shape.as_list()[:3], 1),
                        dtype=tf.float32)

        # convolution way of mean, avg pool will produce [h, w, c]
        # and we need mean of a block [5,5,channel], so we take mean of avg pooled image at channel axis
        # output shape will be (batch, height, width, 1)

        # mean_x = tf.reduce_mean(tf.nn.avg_pool(inputs,
        #                                        ksize=[1, kernel_size[0], kernel_size[1], 1],
        #                                        strides=[1, 1, 1, 1],
        #                                        padding='VALID'), axis=-1, keep_dims=True)

        # or it can be achieved by conv2d with constant weights
        mean_weight = tf.get_variable(name=name + '_mean_weights',
                                      shape=[kernel_size[0], kernel_size[1], inputs.shape.as_list()[-1], 1],
                                      dtype=tf.float32,
                                      initializer=tf.constant_initializer(1.0),
                                      regularizer=None,
                                      trainable=False)
        mean_x = 1. / l * tf.nn.conv2d(inputs, mean_weight, strides=[1, 1, 1, 1],
                                       padding='VALID',
                                       name='mean_' + name)

        # mean for every filter, output shape will be (16,)
        mean_w = tf.reduce_mean(conv_weight, axis=(0, 1, 2), keep_dims=True)
        hout = (2. / l) * conv - mean_w - mean_x

        # if double_threshold:
        hout = double_thresholding(hout, name, double_threshold)
        # else:
        #     hout = tf.nn.relu(0.5 + hout) if with_relu else 0.5 + hout

        return hout
    else:
        # default conv2d implementation
        conv_bias = tf.get_variable(name=name + '_biases',
                                    shape=[filters],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1),
                                    regularizer=None,
                                    trainable=True)

        hout = conv + conv_bias
        hout = tf.nn.relu(hout) if with_relu else hout

        return hout


def fc_ghd(inputs, out_units, name, with_ghd=True, with_relu=True, double_threshold=False):
    if len(inputs.shape) != 2:
        inputs = tf.reshape(inputs, shape=[-1, reduce(lambda x, y: x * y,
                                                      inputs.shape.as_list()[1:],
                                                      1)])

    fc_weight = tf.get_variable(name=name + '_weights',
                                shape=[inputs.shape.as_list()[1], out_units],
                                dtype=tf.float32,
                                initializer=tf.glorot_normal_initializer(1234),
                                regularizer=None,
                                trainable=True)

    if with_ghd:
        # fc version of ghd is easier than convolution
        l = tf.constant(inputs.shape.as_list()[1],
                        dtype=tf.float32)

        mean_x = tf.reduce_mean(inputs, axis=1, keep_dims=True)
        mean_w = tf.reduce_mean(fc_weight, axis=0, keep_dims=True)

        hout = (2. / l) * tf.matmul(inputs, fc_weight) - mean_w - mean_x

        # if double_threshold:
        hout = double_thresholding(hout, name, double_threshold)
        # else:
        #     hout = tf.nn.relu(0.5 + hout) if with_relu else 0.5 + hout

        return hout
    else:
        # default fully connected implementation
        fc_bias = tf.get_variable(name=name + '_biases',
                                  shape=[out_units],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1),
                                  regularizer=None,
                                  trainable=True)

        # fully connected
        hout = tf.matmul(inputs, fc_weight) + fc_bias
        hout = tf.nn.relu(hout) if with_relu else hout
        return hout
