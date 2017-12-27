import tensorflow as tf

from tf_mnist_net_interface import MnistNet


class MnistNetGHD(MnistNet):
    def conv_ghd(self, inputs, filters, kernel_size, name, with_ghd=True, with_relu=True):
        conv_weight = tf.get_variable(name=name + 'weights',
                                      shape=[kernel_size[0], kernel_size[0], inputs.shape.as_list()[-1], filters],
                                      dtype=tf.float32,
                                      initializer=tf.glorot_normal_initializer(4567),
                                      regularizer=None,
                                      trainable=True)

        conv = tf.nn.conv2d(inputs, conv_weight,
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name=name)

        if with_ghd:
            l = tf.constant(reduce(lambda x, y: x * y, conv_weight.shape.as_list()[:3], 1),
                            dtype=tf.float32)

            mean_x = tf.reduce_mean(tf.nn.avg_pool(inputs,
                                                   ksize=[1, kernel_size[0], kernel_size[1], 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding='VALID'), axis=-1, keep_dims=True)

            mean_w = tf.reduce_mean(conv_weight, axis=(0, 1, 2), keep_dims=True)

            hout = (2. / l) * conv - mean_w - mean_x
            hout = tf.nn.relu(0.5 + hout) if with_relu else hout

            return hout
        else:
            conv_bias = tf.get_variable(name=name + '_biases',
                                        shape=[filters],
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.1),
                                        regularizer=None,
                                        trainable=True)

            hout = conv + conv_bias
            hout = tf.nn.relu(hout) if with_relu else hout

            return hout

    def fc_ghd(self, inputs, out_units, name, with_ghd=True, with_relu=True):
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
            l = tf.constant(inputs.shape.as_list()[1],
                            dtype=tf.float32)

            mean_x = tf.reduce_mean(inputs, axis=1, keep_dims=True)
            mean_w = tf.reduce_mean(fc_weight, axis=0, keep_dims=True)

            hout = (2. / l) * tf.matmul(inputs, fc_weight) - mean_w - mean_x
            hout = tf.nn.relu(0.5 + hout) if with_relu else hout

            return hout
        else:
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

    def __init__(self, lr=0.1, batch_size=256, input_shape=None, with_relu=True):
        super(MnistNetGHD, self).__init__(lr, batch_size, input_shape)

        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[None] + input_shape)
        self.labels = tf.placeholder(dtype=tf.int64,
                                     shape=[None, 1])

        self.conv1 = self.conv_ghd(self.inputs, 16, [5, 5], name='conv1', with_ghd=True, with_relu=with_relu)
        self.pool1 = tf.layers.max_pooling2d(self.conv1, [2, 2], [2, 2], name='pool1')
        self.conv2 = self.conv_ghd(self.pool1, 64, [5, 5], name='conv2', with_ghd=True, with_relu=with_relu)
        self.pool2 = tf.layers.max_pooling2d(self.conv2, [2, 2], [2, 2], name='pool2')

        self.fc3 = self.fc_ghd(self.pool2, 1024, 'fc3', with_ghd=True, with_relu=with_relu)
        self.fc4 = self.fc_ghd(self.fc3, 10, 'fc4', with_ghd=True, with_relu=with_relu)

        self.accuracy = tf.metrics.accuracy(labels=self.labels,
                                            predictions=tf.argmax(self.fc4, axis=1))

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels,
                                                           logits=self.fc4)

        self.optimizer = tf.train.AdamOptimizer(lr)

        self.batch_size = batch_size
        self.train_op = self.optimizer.minimize(self.loss)
