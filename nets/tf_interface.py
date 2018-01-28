import sys

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


##############
# Parent Net #
##############

class Net(object):
    def __init__(self, lr=0.1, batch_size=256, input_shape=None, aug=True, zero_mean=False, unit_variance=False):
        self.lr = lr
        self.batch_size = batch_size
        self.input_shape = input_shape

        self.inputs = None
        self.labels = None

        self.train_op = None
        self.loss = None
        self.accuracy = None

        self.aug = aug

        self.train_gen_options = {
            'samplewise_center': zero_mean,
            'samplewise_std_normalization': unit_variance,
            'horizontal_flip': self.aug,
            'vertical_flip': False,
            'rotation_range': 15,
            'width_shift_range': 0.25
        }

        self.val_gen_options = {
            'samplewise_center': zero_mean,
            'samplewise_std_normalization': unit_variance
        }

        self.test_gen_options = {
            'samplewise_center': zero_mean,
            'samplewise_std_normalization': unit_variance
        }

        self.sess = tf.Session()

    def train(self, epochs, X_train, Y_train, X_val, Y_val):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        train_gen = ImageDataGenerator(**self.train_gen_options)
        val_gen = ImageDataGenerator(**self.val_gen_options)
        # train_gen = ImageDataGenerator(width_shift_range=0.1,
        #                                height_shift_range=0.1,
        #                                horizontal_flip=True)
        train_gen = train_gen.flow(X_train, Y_train,
                                   self.batch_size,
                                   seed=123123)
        val_gen = val_gen.flow(X_val, Y_val,
                               self.batch_size,
                               shuffle=False)

        for e in xrange(epochs):
            # indices = np.arange(len(X_train))
            # np.random.seed(np.random.randint(1000000))
            # np.random.shuffle(indices)
            #
            # X_train = X_train[indices]
            # Y_train = Y_train[indices]

            steps = int(len(X_train) / self.batch_size)
            curr_mini_batches = 0
            avgloss = 0
            avgacc = 0
            for s in xrange(steps):
                x, y = train_gen.next()
                # x = X_train[s * self.batch_size:(s + 1) * self.batch_size]
                # y = Y_train[s * self.batch_size:(s + 1) * self.batch_size]

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
                    '\rtraining loss %.5f acc %.5f at %s/%s epoch %s' % (
                        loss, acc, curr_mini_batches, X_train.shape[0], e + 1))
                sys.stdout.flush()

            avgloss /= steps
            avgacc /= steps
            sys.stdout.write(
                '\rtraining loss %.5f acc %.5f at %s/%s epoch %s' % (
                    avgloss, avgacc, curr_mini_batches, X_train.shape[0], e + 1))
            sys.stdout.flush()
            print

            steps = int(len(X_val) / self.batch_size)
            avgloss = 0
            avgacc = 0
            for s in xrange(steps):
                x, y = val_gen.next()
                # x = X_val[s * self.batch_size:(s + 1) * self.batch_size]
                # y = Y_val[s * self.batch_size:(s + 1) * self.batch_size]

                loss, acc = self.sess.run(
                    [self.loss, self.accuracy],
                    feed_dict={
                        self.inputs: x,
                        self.labels: y
                    })

                avgloss += loss
                avgacc += acc

                sys.stdout.write('\rvalidation loss %.5f acc %.5f epoch %s' % (loss, acc, e + 1))
                sys.stdout.flush()

            avgloss /= steps
            avgacc /= steps
            sys.stdout.write('\rvalidation loss %.5f acc %.5f epoch %s' % (avgloss, avgacc, e + 1))
            sys.stdout.flush()
            print

    def evaluate(self, X_test, Y_test):
        gen = ImageDataGenerator(**self.test_gen_options)
        gen = gen.flow(X_test, Y_test,
                       self.batch_size,
                       shuffle=False)
        steps = int(len(X_test) / self.batch_size)
        avgloss = 0
        avgacc = 0
        for s in xrange(steps):
            x, y = gen.next()
            # x = X_test[s * self.batch_size:(s + 1) * self.batch_size]
            # y = Y_test[s * self.batch_size:(s + 1) * self.batch_size]

            loss, acc = self.sess.run(
                [self.loss, self.accuracy],
                feed_dict={
                    self.inputs: x,
                    self.labels: y
                })

            avgloss += loss
            avgacc += acc

            sys.stdout.write('\rtesting loss %.5f acc %.5f' % (loss, acc))
            sys.stdout.flush()

        avgloss /= steps
        avgacc /= steps
        sys.stdout.write('\rtesting loss %.5f acc %.5f' % (avgloss, avgacc))
        sys.stdout.flush()
        print

    def close(self):
        self.sess.close()
