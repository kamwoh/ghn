from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Softmax, Dense, Flatten
from keras.models import Sequential

from keras_layers import ConvGHD, FCGHD


def get_mnist_net(with_ghd, double_threshold=False):
    if with_ghd:
        ghd_model = Sequential()
        ghd_model.add(ConvGHD(filters=16,
                              kernel_size=[5, 5],
                              double_threshold=double_threshold,
                              input_shape=(28, 28, 1),
                              name='conv1'))
        ghd_model.add(MaxPooling2D(pool_size=[2, 2],
                                   strides=[2, 2]))
        ghd_model.add(Dropout(0.5))

        ghd_model.add(ConvGHD(filters=64,
                              kernel_size=[5, 5],
                              double_threshold=double_threshold,
                              name='conv2'))
        ghd_model.add(MaxPooling2D(pool_size=[2, 2],
                                   strides=[2, 2]))
        ghd_model.add(Dropout(0.5))

        ghd_model.add(Flatten())
        ghd_model.add(FCGHD(units=1024,
                            double_threshold=double_threshold,
                            name='fc3'))
        ghd_model.add(Dropout(0.5))
        ghd_model.add(FCGHD(units=10,
                            double_threshold=double_threshold,
                            name='fc4'))
        ghd_model.add(Softmax())
        ghd_model.compile(optimizer=optimizers.Adam(0.1),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        return ghd_model
    else:
        model = Sequential()
        model.add(Conv2D(filters=16,
                         kernel_size=[5, 5],
                         input_shape=(28, 28, 1),
                         activation='relu',
                         name='conv1'))
        model.add(MaxPooling2D(pool_size=[2, 2],
                               strides=[2, 2]))
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=64,
                         kernel_size=[5, 5],
                         activation='relu',
                         name='conv2'))
        model.add(MaxPooling2D(pool_size=[2, 2],
                               strides=[2, 2]))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(units=1024,
                        activation='relu',
                        name='fc3'))
        model.add(Dropout(0.5))
        model.add(Dense(units=10,
                        name='fc4'))
        model.add(Softmax())
        model.compile(optimizer=optimizers.Adam(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
