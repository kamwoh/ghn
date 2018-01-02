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

    X_test = X_test / 255.

    # mean = X_train.mean(axis=(0, 1, 2))
    #
    # X_train -= mean
    # X_val -= mean
    # X_test -= mean

    #############
    # mnist net #
    #############
    net = CifarNetGHD(lr=0.1,
                      batch_size=128,
                      input_shape=[32, 32, 3],
                      with_relu=True,
                      fuzziness_relu=True,
                      nclass=nclass)
    net.train(10, X_train, Y_train, X_val, Y_val)
    net.evaluate(X_test, Y_test)
    net.close()


if __name__ == '__main__':
    main()
