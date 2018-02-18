import cv2
import numpy as np

from nets.keras_models import *
from utils import imageutils, visutils


class MnistImageInput(object):
    def __init__(self):
        from dataset import get_mnist
        self.X_test = get_mnist()[2][0]

    def get_image(self, img_idx):
        img_disp = self.X_test[img_idx]
        img_disp = cv2.resize(img_disp, (28, 28))
        img = img_disp[np.newaxis, :, :, np.newaxis]
        img = img.astype(np.float32)
        return img_disp, img

    def get_length(self):
        return len(self.X_test)

    def has_next_image(self, img_idx):
        return img_idx > 0 or self.get_length() - 1


class CompareModel(object):
    def __init__(self):
        self.models = []

    def add(self, model):
        self.models.append(model)

    def get_activation(self, model_idx, layer_idx):
        pass


def main():
    ghd_model = ghd_mnist_model(0.1, True)
    ghd_model.load_weights('./keras_mnist_ghd.h5')

    naive_model = naive_mnist_model(0.01)
    naive_model.load_weights('./keras_mnist.h5')

    img_input = MnistImageInput()
    img_idx = 0
    layer_idx = 0

    for i, layer in enumerate(ghd_model.layers):
        print(i, layer)

    is_changed = True

    while True:
        if is_changed:
            img_disp, img = img_input.get_image(img_idx)

            is_changed = False
            out = visutils.activation(img, ghd_model, layer_idx)
            out_wo_ghd = visutils.activation(img, naive_model, layer_idx)

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

            out = imageutils.normalize(out)
            disp = imageutils.combine_and_fit(out, is_conv=is_conv, is_fc=is_fc, disp_w=800)

            out_wo_ghd = imageutils.normalize(out_wo_ghd)
            disp_wo_ghd = imageutils.combine_and_fit(out_wo_ghd, is_conv=is_conv, is_fc=is_fc, disp_w=800)

            cv2.imshow('input', img_disp)
            cv2.imshow('disp ghd', disp)
            cv2.imshow('disp_wo_ghd', disp_wo_ghd)

            weight = ghd_model.get_weights()[0]
            weight = imageutils.normalize_weights(weight, 'conv')
            weight = np.transpose(weight, (3, 0, 1, 2))
            weight_disp = imageutils.combine_and_fit(weight, is_weights=True, disp_w=400)

            weight_wo_ghd = naive_model.get_weights()[0]
            weight_wo_ghd = imageutils.normalize_weights(weight_wo_ghd, 'conv')
            weight_wo_ghd = np.transpose(weight_wo_ghd, (3, 0, 1, 2))
            weight_disp_wo_ghd = imageutils.combine_and_fit(weight_wo_ghd, is_weights=True, disp_w=400)

            cv2.imshow('weight_disp_ghd', weight_disp)
            cv2.imshow('weight_disp_wo_ghd', weight_disp_wo_ghd)

            res = ghd_model.predict(img)[0]
            print('output_ghd', np.argmax(res), res.max())

            res = naive_model.predict(img)[0]
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
            if img_input.has_next_image(img_idx):
                img_idx -= 1
                is_changed = True
                print('current img_idx', img_idx)
        elif val == ord('k'):
            if img_input.has_next_image(img_idx):
                img_idx += 1
                is_changed = True
                print('current img_idx', img_idx)


if __name__ == '__main__':
    main()
