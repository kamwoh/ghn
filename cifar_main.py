import numpy as np
from keras.datasets import cifar10, cifar100

from nets.tf_ghd import CifarNetGHD


def main():
    ##############
    # loads data #
    ##############
    nclass = 10

    if nclass == 10:
        cifar = cifar10
    else:
        cifar = cifar100

    (X_train, Y_train), (X_test, Y_test) = cifar.load_data()

    X_train = np.expand_dims(X_train, -1)
    Y_train = np.expand_dims(Y_train, -1)

    X_test = np.expand_dims(X_test, -1) / 255.
    Y_test = np.expand_dims(Y_test, -1)

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

    #############
    # mnist net #
    #############
    net = CifarNetGHD(lr=0.1,
                      batch_size=32,
                      input_shape=[28, 28, 1],
                      with_relu=True,
                      fuzziness_relu=True,
                      nclass=nclass)
    net.train(1, X_train, Y_train, X_val, Y_val)
    net.evaluate(X_test, Y_test)
    net.close()


if __name__ == '__main__':
    main()
