import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Model
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from layers.ghd import GHD, Conv2DGHD, DenseGHD


class MnistNetBN(object):
    def __init__(self, lr=0.1):
        model = Sequential()
        model.add(Conv2D(16, (5, 5), input_shape=(28, 28, 1)))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, (5, 5)))
        model.add(MaxPooling2D())
        model.add(Dense(1024))
        model.add(Dense(10))

        sgd = SGD(lr=lr)
        model.compile(optimizer=sgd,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])


class MnistNetWOBN(object):
    def __init__(self, lr=0.1):
        model = Sequential()
        model.add(Conv2D(16, (5, 5), input_shape=(28, 28, 1)))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, (5, 5)))
        model.add(MaxPooling2D())
        model.add(Dense(1024))
        model.add(Dense(10))

        sgd = SGD(lr=lr)
        model.compile(optimizer=sgd,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])


class MnistNetGHD(object):
    def custom_sparse_categorical_crossentropy(self, target, output):
        # output_shape = output.get_shape()
        # targets = K.cast(K.flatten(target), 'int64')
        targets = tf.cast(tf.reshape(target, [-1]), tf.int64)
        # print targets
        # logits = tf.reshape(output, [-1, int(output_shape[-1])])
        # print target
        # print output
        res = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets,
                logits=output)
        return res

    def __init__(self, lr=0.1, batch_size=32):
        model = Sequential()
        # model.add(Conv2D(16, (5, 5), input_shape=(28, 28, 1)))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(Conv2DGHD(16, (5, 5), (1, 1), input_shape=(28, 28, 1)))
        model.add(MaxPooling2D())
        model.add(Conv2DGHD(64, (5, 5), (1, 1)))
        # model.add(Conv2D(64, (5, 5)))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(GHD(True, False))
        model.add(MaxPooling2D())
        model.add(Flatten(input_shape=(28, 28, 1)))
        model.add(DenseGHD(1024))
        # model.add(DenseGHD(1024))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(GHD(True, False))
        model.add(DenseGHD(10))
        # model.add(GHD(True, False))
        # model.add(BatchNormalization())
        # model.add(Activation('softmax'))
        sgd = Adam(lr=lr)
        model.compile(optimizer=sgd,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        self.model = model
        self.batch_size = batch_size

    def train(self, epochs, X, Y, val_X, val_Y):
        data_generator = ImageDataGenerator(horizontal_flip=True)
        data_generator.fit(X)

        train_gen = data_generator.flow(X, Y, batch_size=self.batch_size)

        data_generator = ImageDataGenerator()

        val_gen = data_generator.flow(val_X, val_Y, batch_size=self.batch_size)

        model_checkpoint = ModelCheckpoint(filepath='./model.h5',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=10,
                                       verbose=1)

        x = get_activations(self.model, np.expand_dims(X[0], 0), print_shape_only=True)
        # print x

        for row in x[-1]:
            for col in row:
                print col
        # print '---'
        # for row in x[1][0]:
        #     for col in row:
        #         print col
        self.model.fit_generator(generator=train_gen,
                                 steps_per_epoch=int(len(X) / self.batch_size),
                                 epochs=epochs,
                                 callbacks=[model_checkpoint, early_stopping],
                                 validation_data=val_gen,
                                 validation_steps=int(len(val_X) / self.batch_size))

        # x = get_activations(self.model, np.expand_dims(X[0], 0), print_shape_only=True)
        # print x
        # for row in x[0][0]:
        #     for col in row:
        #         print col


import keras.backend as K


def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs
    # outputs2 = [K.gradients(layer.output, layer.weights) for layer in model.layers if
    #             layer.name == layer_name or layer_name is None]
    # outputs = []
    # for out in outputs2:
    #     if out != []:
    #         print out
    #         outs = []
    #         for o in out:
    #             outs.append(o * 0.2)
    #         outputs += outs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.expand_dims(X_train, -1)
Y_train = np.expand_dims(Y_train, -1)

split = int(len(X_train) * 0.2)
indices = np.arange(len(X_train))
np.random.seed(np.random.randint(1000000))
np.random.shuffle(indices)

X_train = X_train[indices]
Y_train = Y_train[indices]

X_val = X_train[:split] / 255.
Y_val = Y_train[:split]

X_train = X_train[split:] / 255.
Y_train = Y_train[split:]

net = MnistNetGHD(lr=0.1, batch_size=512)
net.train(10, X_train, Y_train, X_val, Y_val)
