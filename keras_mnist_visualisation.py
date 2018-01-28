import cv2
import numpy as np

import utils


def main():
    from keras_mnist_main import get_mnist_net
    from dataset import get_mnist
    X_test = get_mnist()[2][0]

    ghd_model = get_mnist_net(True)
    ghd_model.load_weights('./keras_mnist_ghd.h5')

    model = get_mnist_net(False)
    model.load_weights('./keras_mnist.h5')

    img_idx = 0
    layer_idx = 0

    for i, layer in enumerate(ghd_model.layers):
        print(i, layer)

    is_changed = True

    while True:
        if is_changed:
            img_disp = X_test[img_idx]
            img_disp = cv2.resize(img_disp, (28, 28))
            img = img_disp[np.newaxis, :, :, np.newaxis]
            img = img.astype(np.float32)

            is_changed = False
            out = utils.activation(img, ghd_model, layer_idx)
            out_wo_ghd = utils.activation(img, model, layer_idx)

            if len(out.shape) == 4:
                is_conv = True
                is_fc = False
                out = np.transpose(out, (3, 1, 2, 0))
                out_wo_ghd = np.transpose(out_wo_ghd, (3, 1, 2, 0))
            else:
                is_conv = False
                is_fc = True
                out = np.transpose(out, (1, 0))
                out_wo_ghd = np.transpose(out_wo_ghd, (1, 0))

            out = utils.normalize(out)
            disp = utils.combine_and_fit(out, is_conv=is_conv, is_fc=is_fc, disp_w=800)

            out_wo_ghd = utils.normalize(out_wo_ghd)
            disp_wo_ghd = utils.combine_and_fit(out_wo_ghd, is_conv=is_conv, is_fc=is_fc, disp_w=800)

            cv2.imshow('input', img_disp)
            cv2.imshow('disp ghd', disp)
            cv2.imshow('disp_wo_ghd', disp_wo_ghd)

            weight = ghd_model.get_weights()[0]
            weight = utils.normalize_weights(weight, 'conv')
            weight = np.transpose(weight, (3, 0, 1, 2))
            weight_disp = utils.combine_and_fit(weight, is_weights=True, disp_w=400)

            weight_wo_ghd = model.get_weights()[0]
            weight_wo_ghd = utils.normalize_weights(weight_wo_ghd, 'conv')
            weight_wo_ghd = np.transpose(weight_wo_ghd, (3, 0, 1, 2))
            weight_disp_wo_ghd = utils.combine_and_fit(weight_wo_ghd, is_weights=True, disp_w=400)

            cv2.imshow('weight_disp_ghd', weight_disp)
            cv2.imshow('weight_disp_wo_ghd', weight_disp_wo_ghd)

            res = ghd_model.predict(img)[0]
            print('output_ghd', np.argmax(res), res.max())

            res = model.predict(img)[0]
            print('output_wo_ghd', np.argmax(res), res.max())

        val = cv2.waitKey(1) & 0xFF

        if val == ord('q'):
            break
        elif val == ord('w'):
            if layer_idx < 8:
                layer_idx += 1
                is_changed = True
                print(ghd_model.layers[layer_idx].name)
        elif val == ord('s'):
            if layer_idx > 0:
                layer_idx -= 1
                is_changed = True
                print(ghd_model.layers[layer_idx].name)
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
