import tensorflow as tf
from tensorflow.contrib import slim

from tf_mnist_net_interface import MnistNet


class MnistNetBN(MnistNet):
    def __init__(self, lr=0.1, batch_size=256, input_shape=None):
        super(MnistNetBN, self).__init__(lr, batch_size, input_shape)

        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[None] + input_shape)
        self.labels = tf.placeholder(dtype=tf.int64,
                                     shape=[None, 1])
        self.is_training = tf.placeholder(dtype=tf.bool)

        scope_bn = slim.arg_scope([slim.batch_norm],
                                  is_training=self.is_training,
                                  activation_fn=tf.nn.relu)
        scope_convfc = slim.arg_scope([slim.conv2d, slim.fully_connected],
                                      weights_initializer=tf.glorot_normal_initializer(1234),
                                      biases_initializer=tf.constant_initializer(0.1),
                                      activation_fn=tf.identity)

        with slim.arg_scope([scope_bn, scope_convfc]):
            self.conv1 = slim.conv2d(self.inputs, 16, [5, 5], padding='VALID')
            self.bn1 = slim.batch_norm(self.conv1)
            self.pool1 = slim.max_pool2d(self.bn1, [2, 2])
            self.conv2 = slim.conv2d(self.pool1, 64, [5, 5], padding='VALID')
            self.bn2 = slim.batch_norm(self.conv2)
            self.pool2 = slim.max_pool2d(self.bn2, [2, 2])

            self.fc3 = slim.fully_connected(self.pool2, 1024)
            self.bn3 = slim.batch_norm(self.fc3)
            self.fc4 = slim.fully_connected(self.bn3, 10)
            self.bn4 = slim.batch_norm(self.fc4)

            self.accuracy = tf.metrics.accuracy(labels=self.labels,
                                                predictions=tf.argmax(self.bn4, axis=1))

            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels,
                                                               logits=self.bn4)

            self.optimizer = tf.train.AdamOptimizer(lr)
            self.train_op = self.optimizer.minimize(self.loss)
