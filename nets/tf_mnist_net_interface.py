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

        self.sess = tf.Session()

    def train(self, epochs, X_train, Y_train, X_val, Y_val):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        for e in xrange(epochs):
            indices = np.arange(len(X_train))
            np.random.seed(np.random.randint(1000000))
            np.random.shuffle(indices)

            X_train = X_train[indices]
            Y_train = Y_train[indices]

            steps = int(len(X_train) / self.batch_size)
            curr_mini_batches = 0
            avgloss = 0
            avgacc = 0
            for s in xrange(steps):
                x = X_train[s * self.batch_size:(s + 1) * self.batch_size]
                y = Y_train[s * self.batch_size:(s + 1) * self.batch_size]

                curr_mini_batches += self.batch_size

                _, loss, acc = self.sess.run(
                    [self.train_op, self.loss, self.accuracy],
                    feed_dict={
                        self.inputs: x,
                        self.labels: y
                    })

                avgloss += loss
                avgacc += acc

                sys.stdout.write(
                    '\rtraining loss %s acc %s at %s/%s' % (loss, acc, curr_mini_batches, X_train.shape[0]))

            avgloss /= steps
            avgacc /= steps
            sys.stdout.write(
                '\rtraining loss %s acc %s at %s/%s' % (avgloss, avgacc, curr_mini_batches, X_train.shape[0]))
            print

            steps = int(len(X_val) / self.batch_size)
            avgloss = 0
            avgacc = 0
            for s in xrange(steps):
                x = X_val[s * self.batch_size:(s + 1) * self.batch_size]
                y = Y_val[s * self.batch_size:(s + 1) * self.batch_size]

                loss, acc = self.sess.run(
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

    def evaluate(self, X_test, Y_test):
        steps = int(len(X_test) / self.batch_size)
        avgloss = 0
        avgacc = 0
        for s in xrange(steps):
            x = X_test[s * self.batch_size:(s + 1) * self.batch_size]
            y = Y_test[s * self.batch_size:(s + 1) * self.batch_size]

            loss, acc = self.sess.run(
                [self.loss, self.accuracy],
                feed_dict={
                    self.inputs: x,
                    self.labels: y
                })

            avgloss += loss
            avgacc += acc

            sys.stdout.write('\rtesting loss %s acc %s' % (loss, acc))

        avgloss /= steps
        avgacc /= steps
        sys.stdout.write('\rtesting loss %s acc %s' % (avgloss, avgacc))
        print

    def close(self):
        self.sess.close()


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(y_pred, y_true)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc
