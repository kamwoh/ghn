import cv2
import numpy as np

from dataset import get_mnist


class MnistImageInput(object):
    def __init__(self):
        self.X_test = get_mnist()[2][0]
        self.mean = None
        self.std = None
        self.img_idx = 0

    def set_mean(self, mean):
        self.mean = mean

    def set_std(self, std):
        self.std = std

    def get_image(self):
        img_disp = self.X_test[self.img_idx]
        img_disp = cv2.resize(img_disp, (28, 28))
        img = img_disp[np.newaxis, :, :, np.newaxis]
        img = img.astype(np.float32)

        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std

        return img_disp, img

    def get_length(self):
        return len(self.X_test)

    def has_next_image(self):
        return self.img_idx > 0 or self.img_idx < self.get_length() - 1
