from keras.callbacks import ModelCheckpoint

import dataset
from nets.keras_models import *


def ghd_main():
    batch_size = 32
    learning_rate = 0.1
    double_threshold = True
    per_pixel = False
    alpha = 0.2

    model = ghd_mnist_model(learning_rate, double_threshold, per_pixel, alpha)
    model.summary()
    for layer in model.layers:
        for weight in layer.trainable_weights:
            print('%s -> %s' % (layer.name, weight.name))

    # train_gen, val_gen, test_gen = dataset.get_mnist_generator(batch_size,
    #                                                            zero_mean=False,
    #                                                            unit_variance=False,
    #                                                            horizontal_flip=True,
    #                                                            rotation_range=10,
    #                                                            width_shift_range=0.1)
    #
    # model.fit_generator(train_gen,
    #                     epochs=5,
    #                     validation_data=val_gen,
    #                     callbacks=[ModelCheckpoint('./keras_mnist_ghd.h5', save_best_only=True)])
    # model.evaluate_generator(test_gen)


def bn_main():
    batch_size = 32
    learning_rate = 0.1

    model = bn_mnist_model(learning_rate)
    model.summary()

    train_gen, val_gen, test_gen = dataset.get_mnist_generator(batch_size,
                                                               zero_mean=False,
                                                               unit_variance=False,
                                                               horizontal_flip=True,
                                                               rotation_range=10,
                                                               width_shift_range=0.1)

    model.fit_generator(train_gen,
                        epochs=5,
                        validation_data=val_gen,
                        callbacks=[ModelCheckpoint('./keras_mnist_bn.h5', save_best_only=True)])
    model.evaluate_generator(test_gen)


def naive_main():
    batch_size = 32
    learning_rate = 0.1

    model = naive_mnist_model(learning_rate)
    model.summary()

    train_gen, val_gen, test_gen = dataset.get_mnist_generator(batch_size,
                                                               zero_mean=False,
                                                               unit_variance=False,
                                                               horizontal_flip=True,
                                                               rotation_range=10,
                                                               width_shift_range=0.1)

    model.fit_generator(train_gen,
                        epochs=5,
                        validation_data=val_gen,
                        callbacks=[ModelCheckpoint('./keras_mnist.h5', save_best_only=True)])
    model.evaluate_generator(test_gen)


if __name__ == '__main__':
    ghd_main()
    # bn_main()
    # naive_main()
