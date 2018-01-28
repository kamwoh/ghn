import math
import sys
import time

import cv2
from keras import backend as K
from keras.models import Model
from skimage.restoration import denoise_tv_bregman

from deepdream_utils import *


def generate_random_image(n, shape):
    return np.random.uniform(size=[n] + shape)


def denoise_images(images):
    tv_denoise_weight = 2.0

    for i, image in enumerate(images):
        min_image = image.min()
        max_image = image.max()
        bgr_image = (image - min_image) / (max_image - min_image)
        rgb_image = bgr_image[:, :, ::-1]
        denoise_rgb = denoise_tv_bregman(rgb_image,
                                         weight=tv_denoise_weight)
        denoise_bgr = denoise_rgb[:, :, ::-1]
        scaled_denoise_bgr = (denoise_bgr * (max_image - min_image) + min_image).reshape(
            images[i].shape)
        images[i] = scaled_denoise_bgr


def deepdream(x_input, model, out_idx, channel, step=1.0, iterations=20,
              octave_n=3, octave_scale=1.4, lap_n=4, g=None, sess=None):
    """

    :type model: Model
    """
    print('computing...')
    start_time = time.time()
    with g.as_default():
        with sess.as_default():
            input_tensor = model.layers[0].input
            out_tensor = model.layers[out_idx].output
            t_objective = K.mean(out_tensor, axis=(0, 1, 2))  # loss
            t_grad = K.gradients(t_objective[channel], input_tensor)[0]

            t_lap_in = K.placeholder(dtype=np.float32, name='lap_in')
            t_laplacian_pyramid = lap_normalize(t_lap_in, lap_n)

            t_simage = K.placeholder(dtype=np.float32, name='image_to_resize')
            t_dsize = K.placeholder(dtype=np.int32, name='size_to_resize')
            t_dimage = tf.image.resize_bilinear(t_simage, t_dsize)

            grad_f = K.function([input_tensor],
                                [t_grad])
            resize_f = K.function([t_simage, t_dsize],
                                  [t_dimage])
            lap_f = K.function([t_lap_in],
                               [t_laplacian_pyramid])

            images = x_input.copy()

            for octave in range(octave_n):
                if octave > 0:
                    hw = np.float32(images.shape[1:3]) * octave_scale
                    images = resize_f([images, np.int32(hw)])[0]
                    denoise_images(images)

                for i in range(iterations):
                    h, w = images.shape[1:3]  # resized hw
                    sz = input_tensor.get_shape().as_list()[1:3]  # original size
                    sx = np.random.randint(sz[1], size=1)
                    sy = np.random.randint(sz[0], size=1)

                    shifted_x = np.roll(images, shift=sx, axis=2)
                    shifted_images = np.roll(shifted_x, shift=sy, axis=1)

                    max_y = max(h - sz[0] // 2, sz[0])
                    jump_y = sz[0]
                    max_x = max(w - sz[1] // 2, sz[1])
                    jump_x = sz[1]

                    grads = np.zeros_like(images)

                    for y in range(0, max_y, jump_y):
                        for x in range(0, max_x, jump_x):
                            feed_images = shifted_images[:, y:y + sz[0], x:x + sz[1]]
                            try:
                                out = grad_f([feed_images])[0]
                                grads[:, y:y + sz[0], x:x + sz[0]] = out
                            except:
                                pass

                    unshifted_x = np.roll(grads, -sx, axis=2)
                    lap_in = np.roll(unshifted_x, -sy, axis=1)
                    lap_out = lap_f([lap_in])[0]

                    images += lap_out

                    yield images

    print('\ntotal time: {} seconds'.format(time.time() - start_time))
    # return x_copy


def activation(x, model, out_idx):
    f = K.function([model.layers[0].input],
                   [model.layers[out_idx].output])
    out = f([x])[0]
    return out


def deconv(x, model, out_idx, batch=8, g=None, sess=None):
    """

    :type model: Model
    """
    print('computing...')
    start_time = time.time()
    with g.as_default():
        with sess.as_default():
            out_tensor = model.layers[out_idx].output
            input_tensor_shape = model.layers[0].input.get_shape().as_list()
            n_filters = out_tensor.get_shape().as_list()[-1]
            out = np.zeros([n_filters] + input_tensor_shape[1:], dtype=np.float32)
            idx = [K.placeholder(dtype=tf.int32) for j in range(batch)]
            gradients = [K.gradients(K.transpose(K.transpose(out_tensor)[idx[j]]),
                                     model.layers[0].input)[0] for j in range(batch)]
            for i in range(0, n_filters, batch):
                sys.stdout.write('\r{}/{}'.format(i // batch + 1, n_filters // batch))
                sys.stdout.flush()
                f = K.function([model.layers[0].input] + idx,
                               gradients)
                _out = f([x] + [i + j for j in range(batch)])
                _out = np.concatenate(_out, axis=0)
                out[i:i + batch] = _out
    print('\ntotal time: {} seconds'.format(time.time() - start_time))
    return out


def visstd(data, s=0.1, per_image=False):
    '''Normalize the image range for visualization'''
    if per_image:
        new_img = np.zeros(data.shape)

        for i in range(data.shape[0]):
            new_img[i] = visstd(data[i])

        return new_img
    else:
        return (data - data.mean()) / max(data.std(), 1e-4) * s + 0.5


def normalize(data, per_image=False):
    if len(data.shape) == 2:
        return normalize_image(data, per_image=False)
    else:
        return normalize_image(data, per_image)


def normalize_image(img, per_image=False):
    if per_image:
        new_img = np.zeros(img.shape)

        for i in range(img.shape[0]):
            new_img[i] = normalize_image(img[i])

        return new_img
    else:
        min_img = img.min()
        max_img = img.max()
        return (img - min_img) / (max_img - min_img + 1e-7)


def normalize_weights(weights, mode):
    if mode == 'conv':
        if weights.shape[2] == 3:  # (h, w, c, n_filter)
            new_weights = normalize_image(weights)
        else:
            new_weights = np.zeros(weights.shape)
            for i in range(weights.shape[3]):
                new_weights[..., i] = normalize_image(weights[..., i])
        return new_weights


def to_255(img):
    img = img * 255.0
    img = img.astype(np.uint8)
    return img


def rgb_to_bgr(img):
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        return img[:, :, ::-1]
    else:
        return img[:, :, :, ::-1]


def combine_and_fit(data, gap=1, is_conv=False, is_fc=False, is_deconv=False, is_weights=False, disp_w=800):
    if len(data.shape) == 4:
        h, w = data.shape[1:3]
        if disp_w is None:
            disp_w = data.shape[0] * w  # default shape
    else:
        h, w = 1, 1

    total = len(data)
    if is_deconv or is_weights:
        n_col = int(math.ceil(math.sqrt(total)))
        n_col_gap = n_col - 1
        factor = (disp_w - n_col_gap * gap) / float(n_col * w)
        width = int(n_col * w * factor + n_col_gap * gap)
        height = int(n_col * h * factor + n_col_gap * gap)
        y_jump = int(h * factor + gap)
        x_jump = int(w * factor + gap)
        new_h = int(h * factor)
        new_w = int(w * factor)
        img = np.zeros((height, width, 3), dtype=np.float32)
        img += 0.1
        i = 0
        for y in range(n_col):
            y = y * y_jump
            for x in range(n_col):
                if i >= total:
                    break

                x = x * x_jump
                to_y = y + int(h * factor)
                to_x = x + int(w * factor)
                img[y:to_y, x:to_x] = cv2.resize(data[i], (new_w, new_h),
                                                 interpolation=cv2.INTER_AREA)
                i += 1
        return img
    else:
        if is_conv or is_fc:
            n_row = int(math.ceil(math.sqrt(total)))
            n_col = n_row
        else:
            n_row = total
            n_col = data.shape[-1]

        n_col_gap = n_col - 1
        n_row_gap = n_row - 1

        factor = (disp_w - n_col_gap * gap) / float(n_col * w)
        width = int(n_col * w * factor + n_col_gap * gap)
        height = int(n_row * h * factor + n_row_gap * gap)
        y_jump = int(h * factor + gap)
        x_jump = int(w * factor + gap)
        new_w = int(w * factor)
        new_h = int(h * factor)

        img = np.zeros((height, width), dtype=np.float32)
        img += 0.2

        i = 0
        for y in range(n_row):
            y = y * y_jump
            j = 0
            for x in range(n_col):
                x = x * x_jump

                if i >= total:
                    break

                if is_conv:
                    d = data[i, :, :]
                elif is_fc:
                    d = data[i]
                else:
                    d = data[i, :, :, j]

                to_y = y + int(h * factor)
                to_x = x + int(w * factor)
                img[y:to_y, x:to_x] = cv2.resize(d, (new_w, new_h),
                                                 interpolation=cv2.INTER_AREA)

                if is_conv or is_fc:
                    i += 1
                else:
                    j += 1

            if not is_conv and not is_fc:
                i += 1
        return img
