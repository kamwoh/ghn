import cv2
import numpy as np

import utils


def main():
    from keras_mnist_main import get_mnist_net
    from dataset import get_mnist
    X_test = get_mnist()[2][0]

    model = get_mnist_net(True)
    model.load_weights('./keras_mnist_ghd.h5')

    img_idx = 0
    layer_idx = 0

    weight_idx = {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 1,
        5: 2,
        6: 2,
        7: 3,
        8: 3
    }

    for i, layer in enumerate(model.layers):
        print(i, layer)

    is_changed = True

    while True:
        if is_changed:
            img_disp = X_test[img_idx]
            img_disp = cv2.resize(img_disp, (28, 28))
            img = img_disp[np.newaxis, :, :, np.newaxis]
            img = img.astype(np.float32)

            is_changed = False
            out = utils.activation(img, model, layer_idx)
            if len(out.shape) == 4:
                is_conv = True
                is_fc = False
                out = np.transpose(out, (3, 1, 2, 0))
            else:
                is_conv = False
                is_fc = True
                out = np.transpose(out, (1, 0))

            out = utils.normalize(out)
            disp = utils.combine_and_fit(out, is_conv=is_conv, is_fc=is_fc, disp_w=800)

            cv2.imshow('input', img_disp)
            cv2.imshow('disp', disp)

            weight = model.get_weights()[0]
            weight = utils.normalize_weights(weight, 'conv')
            weight = np.transpose(weight, (3, 0, 1, 2))
            weight_disp = utils.combine_and_fit(weight, is_weights=True, disp_w=400)
            cv2.imshow('weight_disp', weight_disp)

        val = cv2.waitKey(1) & 0xFF

        if val == ord('q'):
            break
        elif val == ord('w'):
            if layer_idx < 8:
                layer_idx += 1
                is_changed = True
                print(model.layers[layer_idx].name)
        elif val == ord('s'):
            if layer_idx > 0:
                layer_idx -= 1
                is_changed = True
                print(model.layers[layer_idx].name)
        elif val == ord('i'):
            if img_idx > 0:
                img_idx -= 1
                is_changed = True
                print('current img_idx', img_idx)
        elif val == ord('k'):
            if img_idx < len(X_test) - 1:
                img_idx += 1
                is_changed = True
                print('current img_idx', img_idx)


if __name__ == '__main__':
    main()
