import tensorflow as tf

from tf_interface import Net
from tf_layers import conv_ghd, fc_ghd, accuracy


class MnistNetGHD(Net):
    def __init__(self, lr=0.1, batch_size=256, input_shape=None, aug=False, zero_mean=False, unit_variance=False,
                 with_relu=True, double_threshold=False):
        super(MnistNetGHD, self).__init__(lr, batch_size, input_shape, aug, zero_mean, unit_variance)

        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[None] + input_shape)
        self.labels = tf.placeholder(dtype=tf.int64,
                                     shape=[None, 1])

        self.conv1 = conv_ghd(self.inputs, 16, [5, 5], name='conv1', with_ghd=True, with_relu=with_relu,
                              double_threshold=double_threshold)
        self.pool1 = tf.layers.max_pooling2d(self.conv1, [2, 2], [2, 2], name='pool1')
        self.conv2 = conv_ghd(self.pool1, 64, [5, 5], name='conv2', with_ghd=True, with_relu=with_relu,
                              double_threshold=double_threshold)
        self.pool2 = tf.layers.max_pooling2d(self.conv2, [2, 2], [2, 2], name='pool2')

        self.fc3 = fc_ghd(self.pool2, 1024, 'fc3', with_ghd=True, with_relu=with_relu,
                          double_threshold=double_threshold)
        self.fc4 = fc_ghd(self.fc3, 10, 'fc4', with_ghd=True, with_relu=with_relu, double_threshold=double_threshold)

        self.accuracy = accuracy(y_true=tf.squeeze(self.labels, axis=1),
                                 y_pred=tf.argmax(self.fc4, axis=1))

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels,
                                                           logits=self.fc4)

        self.optimizer = tf.train.AdamOptimizer(lr)

        self.aug = aug
        self.batch_size = batch_size
        self.train_op = self.optimizer.minimize(self.loss)


class CifarNetGHD(Net):
    def __init__(self, lr=0.1, batch_size=256, input_shape=None, aug=False, zero_mean=False, unit_variance=False,
                 with_relu=True, double_threshold=False, nclass=10):
        super(CifarNetGHD, self).__init__(lr, batch_size, input_shape, aug, zero_mean, unit_variance)

        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[None] + input_shape)
        self.labels = tf.placeholder(dtype=tf.int64,
                                     shape=[None, 1])

        self.conv1 = conv_ghd(self.inputs, 64, [3, 3], 'conv1', with_ghd=True, with_relu=with_relu,
                              double_threshold=double_threshold)
        self.conv2 = conv_ghd(self.conv1, 256, [5, 5], 'conv2', with_ghd=True, with_relu=with_relu,
                              double_threshold=double_threshold)
        self.pool2 = tf.layers.max_pooling2d(self.conv2, [2, 2], [2, 2], name='pool2')
        self.conv3 = conv_ghd(self.pool2, 256, [5, 5], 'conv3', with_ghd=True, with_relu=with_relu,
                              double_threshold=double_threshold)
        self.pool3 = tf.layers.max_pooling2d(self.conv3, [2, 2], [2, 2], name='pool3')
        self.fc4 = fc_ghd(self.pool3, 1024, 'fc4', with_ghd=True, with_relu=with_relu,
                          double_threshold=double_threshold)
        self.fc5 = fc_ghd(self.fc4, 512, 'fc5', with_ghd=True, with_relu=with_relu, double_threshold=double_threshold)
        self.fc6 = fc_ghd(self.fc5, nclass, 'fc6', with_ghd=True, with_relu=with_relu,
                          double_threshold=double_threshold)

        self.accuracy = accuracy(y_true=tf.squeeze(self.labels, axis=1),
                                 y_pred=tf.argmax(self.fc6, axis=1))

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels,
                                                           logits=self.fc6)

        self.optimizer = tf.train.AdamOptimizer(lr)

        self.batch_size = batch_size
        self.aug = aug
        self.train_op = self.optimizer.minimize(self.loss)
