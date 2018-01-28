import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


def get_mnist_generator(batch_size=32,
                        zero_mean=False, unit_variance=False,
                        horizontal_flip=True, vertical_flip=False,
                        rotation_range=10,
                        height_shift_range=0, width_shift_range=0.25):
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = get_mnist()

    train_gen = ImageDataGenerator(featurewise_center=zero_mean,
                                   featurewise_std_normalization=unit_variance,
                                   width_shift_range=width_shift_range,
                                   height_shift_range=height_shift_range,
                                   rotation_range=rotation_range,
                                   horizontal_flip=horizontal_flip,
                                   vertical_flip=vertical_flip)
    train_gen.fit(X_train,
                  seed=1234)

    valtest_gen = ImageDataGenerator(featurewise_center=zero_mean,
                                     featurewise_std_normalization=unit_variance)
    valtest_gen.mean = train_gen.mean
    valtest_gen.std = train_gen.std

    train_gen = train_gen.flow(X_train, Y_train,
                               batch_size=batch_size,
                               seed=999)
    val_gen = valtest_gen.flow(X_val, Y_val,
                               batch_size=batch_size,
                               seed=888)
    test_gen = valtest_gen.flow(X_test, Y_test,
                                batch_size=batch_size,
                                seed=777)
    return train_gen, val_gen, test_gen


def get_mnist(split_ratio=0.2):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = np.expand_dims(X_train, -1)
    Y_train = np.expand_dims(Y_train, -1)

    X_test = np.expand_dims(X_test, -1)
    Y_test = to_categorical(np.expand_dims(Y_test, -1), 10)

    split = int(len(X_train) * split_ratio)
    indices = np.arange(len(X_train))
    np.random.seed(np.random.randint(1000000))
    np.random.shuffle(indices)

    X_train = X_train[indices]
    Y_train = to_categorical(Y_train[indices], 10)

    X_val = X_train[:split]
    Y_val = Y_train[:split]

    X_train = X_train[split:]
    Y_train = Y_train[split:]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
