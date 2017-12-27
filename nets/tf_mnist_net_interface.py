import sys

import numpy as np
import tensorflow as tf


class MnistNet(object):
    def __init__(self, lr=0.1, batch_size=256, input_shape=None):
        self.lr = lr
        self.batch_size = batch_size
        self.input_shape = input_shape

        self.inputs = None
        self.labels = None

        self.train_op = None
        self.loss = None
        self.accuracy = None

    def train(self, epochs, X, Y, val_X, val_Y):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for e in xrange(epochs):
                indices = np.arange(len(X))
                np.random.seed(np.random.randint(1000000))
                np.random.shuffle(indices)

                X = X[indices]
                Y = Y[indices]

                steps = int(len(X) / self.batch_size)
                curr_mini_batches = 0
                avgloss = 0
                avgacc = 0
                for s in xrange(steps):
                    x = X[s * self.batch_size:(s + 1) * self.batch_size]
                    y = Y[s * self.batch_size:(s + 1) * self.batch_size]

                    curr_mini_batches += self.batch_size

                    _, loss, acc = sess.run(
                        [self.train_op, self.loss, self.accuracy],
                        feed_dict={
                            self.inputs: x,
                            self.labels: y
                        })

                    avgloss += loss
                    avgacc += acc

                    sys.stdout.write(
                        '\rtraining loss %s acc %s at %s/%s' % (loss, acc, curr_mini_batches, X.shape[0]))

                avgloss /= steps
                avgacc /= steps
                sys.stdout.write(
                    '\rtraining loss %s acc %s at %s/%s' % (avgloss, avgacc, curr_mini_batches, X.shape[0]))
                print

                steps = int(len(val_X) / self.batch_size)
                avgloss = 0
                avgacc = 0
                for s in xrange(steps):
                    x = val_X[s * self.batch_size:(s + 1) * self.batch_size]
                    y = val_Y[s * self.batch_size:(s + 1) * self.batch_size]

                    loss, acc = sess.run(
                        [self.loss, self.accuracy],
                        feed_dict={
                            self.inputs: x,
                            self.labels: y
                        })

                    avgloss += loss
                    avgacc += acc

                    sys.stdout.write('\rvalidation loss %s acc %s' % (loss, acc))

                avgloss /= steps
                avgacc /= steps
                sys.stdout.write('\rvalidation loss %s acc %s' % (avgloss, avgacc))
                print

def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(y_pred, y_true)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc