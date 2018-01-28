from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, MaxPooling2D, Softmax, Conv2D, Dense
from keras.models import Sequential

import dataset
from nets.keras_layers import ConvGHD, FCGHD


def main():
    batch_size = 256
    double_threshold = False

    ghd_model = Sequential()
    ghd_model.add(ConvGHD(filters=16,
                          kernel_size=[5, 5],
                          double_threshold=double_threshold,
                          input_shape=(28, 28, 1),
                          name='conv1'))
    ghd_model.add(MaxPooling2D(pool_size=[2, 2],
                               strides=[2, 2]))
    ghd_model.add(ConvGHD(filters=64,
                          kernel_size=[5, 5],
                          double_threshold=double_threshold,
                          name='conv2'))
    ghd_model.add(MaxPooling2D(pool_size=[2, 2],
                               strides=[2, 2]))

    ghd_model.add(Flatten())
    ghd_model.add(FCGHD(units=1024,
                        double_threshold=double_threshold,
                        name='fc3'))
    ghd_model.add(FCGHD(units=10,
                        double_threshold=double_threshold,
                        name='fc4'))
    ghd_model.add(Softmax())
    ghd_model.compile(optimizer=optimizers.Adam(0.1),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    ghd_model.summary()

    train_gen, val_gen, test_gen = dataset.get_mnist_generator(batch_size,
                                                               zero_mean=False,
                                                               unit_variance=False,
                                                               horizontal_flip=False,
                                                               rotation_range=0,
                                                               width_shift_range=0)

    ghd_model.fit_generator(train_gen,
                            epochs=5,
                            validation_data=val_gen,
                            callbacks=[ModelCheckpoint('./keras_mnist_ghd.h5', save_best_only=True)])

    batch_size = 256

    model = Sequential()
    model.add(Conv2D(filters=16,
                     kernel_size=[5, 5],
                     input_shape=(28, 28, 1),
                     name='conv1'))
    model.add(MaxPooling2D(pool_size=[2, 2],
                           strides=[2, 2]))
    model.add(Conv2D(filters=64,
                     kernel_size=[5, 5],
                     name='conv2'))
    model.add(MaxPooling2D(pool_size=[2, 2],
                           strides=[2, 2]))

    model.add(Flatten())
    model.add(Dense(units=1024,
                    name='fc3'))
    model.add(Dense(units=10,
                    name='fc4'))
    model.add(Softmax())
    model.compile(optimizer=optimizers.Adam(0.1),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    train_gen, val_gen, test_gen = dataset.get_mnist_generator(batch_size,
                                                               zero_mean=False,
                                                               unit_variance=False,
                                                               horizontal_flip=False,
                                                               rotation_range=0,
                                                               width_shift_range=0)

    model.fit_generator(train_gen,
                        epochs=5,
                        validation_data=val_gen,
                        callbacks=[ModelCheckpoint('./keras_mnist.h5', save_best_only=True)])


if __name__ == '__main__':
    main()
