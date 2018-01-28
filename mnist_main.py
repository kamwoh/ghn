import numpy as np
from keras.datasets import mnist

from nets.tf_ghd import MnistNetGHD


def main():
    ##############
    # loads data #
    ##############
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = np.expand_dims(X_train, -1)
    Y_train = np.expand_dims(Y_train, -1)

    X_test = np.expand_dims(X_test, -1)
    Y_test = np.expand_dims(Y_test, -1)

    split = int(len(X_train) * 0.2)
    indices = np.arange(len(X_train))
    np.random.seed(np.random.randint(1000000))
    np.random.shuffle(indices)

    X_train = X_train[indices]
    Y_train = Y_train[indices]

    X_val = X_train[:split]
    Y_val = Y_train[:split]

    X_train = X_train[split:]
    Y_train = Y_train[split:]

    #############
    # mnist net #
    #############
    net = MnistNetGHD(lr=0.1,
                      batch_size=64,
                      input_shape=[28, 28, 1],
                      aug=False,
                      zero_mean=False,
                      unit_variance=False,
                      double_threshold=False)
    net.train(1, X_train, Y_train, X_val, Y_val)
    net.evaluate(X_test, Y_test)
    net.close()


if __name__ == '__main__':
    main()
