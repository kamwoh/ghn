import tensorflow as tf
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Softmax, Dense, Flatten, Activation, BatchNormalization
from keras.models import Sequential

from keras_layers import ConvGHD, FCGHD


def ghd_mnist_model(learning_rate, double_threshold, per_pixel, alpha, relu):
    with tf.variable_scope('ghn'):
        model = Sequential()
        model.add(ConvGHD(filters=16,
                          kernel_size=[5, 5],
                          double_threshold=double_threshold,
                          per_pixel=per_pixel,
                          alpha=alpha,
                          relu=relu,
                          input_shape=(28, 28, 1),
                          name='conv1'))
        model.add(MaxPooling2D(pool_size=[2, 2],
                               strides=[2, 2]))
        model.add(ConvGHD(filters=64,
                          kernel_size=[5, 5],
                          double_threshold=double_threshold,
                          per_pixel=per_pixel,
                          alpha=alpha,
                          relu=relu,
                          name='conv2'))
        model.add(MaxPooling2D(pool_size=[2, 2],
                               strides=[2, 2]))
        model.add(Flatten())
        model.add(FCGHD(units=1024,
                        double_threshold=double_threshold,
                        per_pixel=per_pixel,
                        alpha=alpha,
                        relu=relu,
                        name='fc3'))
        model.add(Dropout(0.5))
        model.add(FCGHD(units=10,
                        double_threshold=double_threshold,
                        per_pixel=per_pixel,
                        alpha=alpha,
                        relu=relu,
                        name='fc4'))
        model.add(Softmax())
        model.compile(optimizer=optimizers.Adam(learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model


def naive_mnist_model(learning_rate):
    with tf.variable_scope('naive'):
        model = Sequential()
        model.add(Conv2D(filters=16,
                         kernel_size=[5, 5],
                         input_shape=(28, 28, 1),
                         name='conv1'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=[2, 2],
                               strides=[2, 2]))
        model.add(Conv2D(filters=64,
                         kernel_size=[5, 5],
                         name='conv2'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=[2, 2],
                               strides=[2, 2]))
        model.add(Flatten())
        model.add(Dense(units=1024,
                        name='fc3'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=10,
                        name='fc4'))
        model.add(Softmax())
        model.compile(optimizer=optimizers.Adam(learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model


def bn_mnist_model(learning_rate):
    with tf.variable_scope('bn'):
        model = Sequential()
        model.add(Conv2D(filters=16,
                         kernel_size=[5, 5],
                         input_shape=(28, 28, 1),
                         name='conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=[2, 2],
                               strides=[2, 2]))
        model.add(Conv2D(filters=64,
                         kernel_size=[5, 5],
                         name='conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=[2, 2],
                               strides=[2, 2]))
        model.add(Flatten())
        model.add(Dense(units=1024,
                        name='fc3'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=10,
                        name='fc4'))
        model.add(BatchNormalization())
        model.add(Softmax())
        model.compile(optimizer=optimizers.Adam(learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model
