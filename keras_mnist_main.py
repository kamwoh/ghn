from keras import optimizers
from keras.layers import Flatten, MaxPooling2D, Softmax
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
                        double_threshold=double_threshold))
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
                            validation_data=val_gen)


if __name__ == '__main__':
    main()
