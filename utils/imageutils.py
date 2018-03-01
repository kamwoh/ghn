import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visstd(data, s=0.1, per_image=False):
    '''Normalize the image range for visualization'''
    if per_image:
        new_img = np.zeros(data.shape)

        for i in range(data.shape[0]):
            new_img[i] = visstd(data[i])

        return new_img
    else:
        return (data - data.mean()) / max(data.std(), 1e-4) * s + 0.5


def normalize(data, per_image=False, heatmap=False):
    if len(data.shape) == 2:
        return normalize_image(data, False, heatmap)
    else:
        return normalize_image(data, per_image, heatmap)


def normalize_image(img, per_image, heatmap):
    if per_image:
        if heatmap:
            new_img = np.zeros(list(img.shape[:3]) + [3])
        else:
            new_img = np.zeros(img.shape)

        for i in range(img.shape[0]):
            new_img[i] = normalize_single_image(img[i], heatmap)

        return new_img
    else:
        return normalize_single_image(img, heatmap)


def normalize_single_image(img, heatmap):
    min_pixel = img.min()
    max_pixel = img.max()
    range_pixel = max_pixel - min_pixel + 1e-9
    print('min -> %s, max -> %s, range -> %s' % (min_pixel, max_pixel, range_pixel))
    out = (img - min_pixel) / range_pixel
    return to_heatmap(out) if heatmap else out


def normalize_weights(weights, mode, per_image, heatmap, first_layer):
    if mode == 'conv':
        if first_layer:
            return normalize_first_layer_conv_weights(weights, per_image, heatmap)
        else:
            return normalize_conv_weights(weights, per_image, heatmap)


def normalize_first_layer_conv_weights(weights, per_image, heatmap):
    if per_image:
        h, w, c, n_filter = weights.shape
        c = 3 if heatmap else c
        new_weights = np.zeros((h, w, c, n_filter))
        for i in range(weights.shape[3]):
            new_weights[..., i] = normalize_image(weights[..., i], False, heatmap)
    else:
        normalized_weights = normalize_image(weights, False, False)
        if heatmap:
            h, w, c, n_filter = normalized_weights.shape
            c = 3 if heatmap else c
            new_weights = np.zeros((h, w, c, n_filter))
            for i in range(weights.shape[3]):
                new_weights[..., i] = to_heatmap(normalized_weights[..., i])
        else:
            new_weights = normalized_weights

    return new_weights


def normalize_conv_weights(weights, per_image, heatmap):
    # output -> h,w,c,n -> h,w,c,n,3 if heatmap
    if per_image:
        h, w, c, n_filter = weights.shape
        new_weights = np.zeros((h, w, c, n_filter))  # h,w,c,n
        for i in range(weights.shape[3]):
            new_weights[..., i] = normalize_image(weights[..., i], False, False)
    else:
        new_weights = normalize_image(weights, False, False)

    if heatmap:
        new_weights = heatmap_on_conv_weights(new_weights)
    else:
        new_weights = np.expand_dims(new_weights, 4)  # to indicate it is 2nd conv and above
    return new_weights


def heatmap_on_conv_weights(weights):
    h, w, c, n_filter = weights.shape
    heatmap_weights = np.zeros((h, w, c, n_filter, 3))
    for i in range(weights.shape[3]):
        for j in range(weights.shape[2]):
            heatmap_weight = to_heatmap(weights[..., j, i])
            for c in range(3):
                heatmap_weights[..., j, i, c] = heatmap_weight[..., c]
    return heatmap_weights


def to_255(img):
    img = img * 255.0
    img = img.astype(np.uint8)
    return img


def to_heatmap(img):
    if len(img.shape) == 4:
        new_img = np.zeros(tuple(list(img.shape[:3]) + [3]), np.float32)
        for i in range(len(img)):
            new_img[i] = to_heatmap(img[i])
        return new_img
    else:
        if len(img.shape) == 3:
            if img.dtype != np.uint8:
                img = to_255(img)

            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = np.squeeze(img, 2)

            img = plt.cm.coolwarm(img)
            img = to_255(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            return img / 255.0
        elif len(img.shape) == 2:
            h, w = img.shape
            if h != w:  # fc layer
                new_img = np.zeros((img.shape[0], 1, 1, img.shape[1]))
                for i in range(img.shape[0]):
                    new_img[i] = to_heatmap(img[i])
                return new_img
            else:
                return to_heatmap(np.expand_dims(img, 2))
        else:
            return img


def combine_and_fit(data, gap=2, is_deconv=False, is_weights=False, disp_w=200):
    first_layer = len(data.shape) != 5
    return fit_data(data, disp_w, gap, first_layer, is_weights or is_deconv)


def fit_data(data, disp_w, gap, first_layer, is_weights):
    if first_layer:
        total, h, w, c = data.shape
    else:
        total, h, w, c, heatmap_n = data.shape

    if not first_layer:
        total *= c

    n_row = int(math.ceil(math.sqrt(total)))
    n_col = n_row

    n_col_gap = n_col - 1
    n_row_gap = n_row - 1

    factor = (disp_w - n_col_gap * gap) / float(n_col * w)
    width = int(n_col * w * factor + n_col_gap * gap)
    height = int(n_row * h * factor + n_row_gap * gap)
    y_jump = int(h * factor + gap)
    x_jump = int(w * factor + gap)
    new_h = int(h * factor)
    new_w = int(w * factor)

    if data.shape[-1] == 3:
        img = np.zeros((height, width, 3), dtype=np.float32)
    else:
        img = np.zeros((height, width), dtype=np.float32)

    img += 0.2
    i = 0

    input_channel_count = 1
    random_shape = () if len(img.shape) == 2 else (3)
    random_color = np.random.random(random_shape)

    for y in range(n_col):
        y = y * y_jump
        for x in range(n_col):
            if i >= total:
                break

            x = x * x_jump
            to_y = y + int(h * factor)
            to_x = x + int(w * factor)

            if first_layer or not is_weights:
                img[y:to_y, x:to_x] = cv2.resize(data[i], (new_w, new_h),
                                                 interpolation=cv2.INTER_AREA)
                i += 1
            else:
                img[y:to_y, x:to_x] = cv2.resize(data[i, :, :, input_channel_count - 1], (new_w, new_h),
                                                 interpolation=cv2.INTER_AREA)
                img[to_y:to_y + 3, x:to_x] = random_color
                img[y:to_y, to_x:to_x + 3] = random_color

                if input_channel_count % c == 0:
                    random_color = np.random.random(random_shape)
                    input_channel_count = 0
                    i += 1

                input_channel_count += 1
    return img
