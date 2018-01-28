from keras.callbacks import ModelCheckpoint

import dataset
from nets.keras_models import get_mnist_net


def main():
    batch_size = 256
    double_threshold = False

    ghd_model = get_mnist_net(True, double_threshold)
    ghd_model.summary()

    train_gen, val_gen, test_gen = dataset.get_mnist_generator(batch_size,
                                                               zero_mean=False,
                                                               unit_variance=False,
                                                               horizontal_flip=False,
                                                               rotation_range=0,
                                                               width_shift_range=0)

    ghd_model.fit_generator(train_gen,
                            epochs=20,
                            validation_data=val_gen,
                            callbacks=[ModelCheckpoint('./keras_mnist_ghd.h5', save_best_only=True)])

    batch_size = 256

    model = get_mnist_net(False)
    model.summary()

    train_gen, val_gen, test_gen = dataset.get_mnist_generator(batch_size,
                                                               zero_mean=False,
                                                               unit_variance=False,
                                                               horizontal_flip=False,
                                                               rotation_range=0,
                                                               width_shift_range=0)

    model.fit_generator(train_gen,
                        epochs=20,
                        validation_data=val_gen,
                        callbacks=[ModelCheckpoint('./keras_mnist.h5', save_best_only=True)])


if __name__ == '__main__':
    main()
