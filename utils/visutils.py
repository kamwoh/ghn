import sys
import time

from keras import backend as K
from keras.models import Model
from skimage.restoration import denoise_tv_bregman

from deepdream_utils import *


def generate_random_image(n, shape):
    return np.random.uniform(size=[n] + shape)


def denoise_images(images):
    tv_denoise_weight = 2.0

    for i, image in enumerate(images):
        min_pixel = image.min()
        max_pixel = image.max()
        bgr_image = (image - min_pixel) / (max_pixel - min_pixel)
        rgb_image = bgr_image[:, :, ::-1]
        denoise_rgb = denoise_tv_bregman(rgb_image,
                                         weight=tv_denoise_weight)
        denoise_bgr = denoise_rgb[:, :, ::-1]
        scaled_denoise_bgr = (denoise_bgr * (max_pixel - min_pixel) + min_pixel).reshape(
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
