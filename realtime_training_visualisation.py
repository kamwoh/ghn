import threading
from collections import OrderedDict

import cv2

import dataset
from gui.window import LayerChoices
from nets.keras_models import *
from utils.datautils import MnistImageInput
from utils.experimentutils import RealtimeModel
from utils.miscutils import get_current_time_in_string


def to_dict(layers):
    ret = OrderedDict()
    for i, layer_name in enumerate(layers):
        ret[layer_name] = i
    return ret


def visualise_thread(img_input, realtime_model):
    """

    :param img_input:
    :type realtime_model: RealtimeModel
    :return:
    """
    with realtime_model.graph.as_default():
        with realtime_model.sess.as_default():
            realtime_model.is_changed = True

            while True:
                if realtime_model.is_changed:
                    realtime_model.is_changed = False

                    img_disp, img = img_input.get_image()
                    cv2.imshow('input', img_disp)

                    realtime_model.visualise(img)

                val = cv2.waitKey(1) & 0xFF

                if val == ord('q'):
                    cv2.destroyAllWindows()
                    break
                elif val == ord('i'):
                    if img_input.has_next_image():
                        img_input.img_idx -= 1
                        realtime_model.is_changed = True
                        print('current img_idx', img_input.img_idx)
                elif val == ord('k'):
                    if img_input.has_next_image():
                        img_input.img_idx += 1
                        realtime_model.is_changed = True
                        print('current img_idx', img_input.img_idx)
                elif val == ord('1'):
                    realtime_model.activation_per_filter = not realtime_model.activation_per_filter
                    realtime_model.is_changed = True
                elif val == ord('2'):
                    realtime_model.activation_heatmap = not realtime_model.activation_heatmap
                    realtime_model.is_changed = True
                elif val == ord('3'):
                    realtime_model.weight_per_filter = not realtime_model.weight_per_filter
                    realtime_model.is_changed = True
                elif val == ord('4'):
                    realtime_model.weight_heatmap = not realtime_model.weight_heatmap
                    realtime_model.is_changed = True


def filter_conv(old_dict, model):
    new_dict = OrderedDict()
    for weight_name, idx in old_dict.items():
        if len(model.trainable_weights[idx].shape) == 4:  # only conv weight
            print(weight_name, idx)
            new_dict[weight_name] = idx
    return new_dict


def main():
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            batch_size = 64
            learning_rate = 0.1

            double_threshold = True
            per_pixel = False
            alpha = 0.2
            ghd_model = ghd_mnist_model(learning_rate, double_threshold, per_pixel, alpha)
            # if os.path.exists('./keras_mnist_ghd.h5'):
            #     ghd_model.load_weights('./keras_mnist_ghd.h5')

            # bn_model = bn_mnist_model(learning_rate)
            # if os.path.exists('./keras_mnist_bn.h5'):
            #     bn_model.load_weights('./keras_mnist_bn.h5')

            # naive_model = naive_mnist_model(learning_rate / 1000.)
            # if os.path.exists('./keras_mnist_naive.h5'):
            #     naive_model.load_weights('./keras_mnist_naive.h5')

            train_gen, val_gen, test_gen, (mean, std) = dataset.get_mnist_generator(batch_size,
                                                                                    zero_mean=True,
                                                                                    unit_variance=True,
                                                                                    horizontal_flip=True,
                                                                                    rotation_range=10,
                                                                                    width_shift_range=0.1)

            img_input = MnistImageInput()
            img_input.set_mean(mean)
            img_input.set_std(std)

            ghd_layers = [layer.name for layer in ghd_model.layers]
            ghd_weights = [weight.name for weight in ghd_model.trainable_weights]
            ghd_layers_dict = to_dict(ghd_layers)
            ghd_weights_dict = filter_conv(to_dict(ghd_weights), ghd_model)
            ghd_weights = map(lambda weight: weight.name,
                              filter(lambda weight: len(weight.shape) == 4, ghd_model.trainable_weights))

            # bn_layers = [layer.name for layer in bn_model.layers]
            # bn_weights = [weight.name for weight in bn_model.trainable_weights]
            # bn_layers_dict = to_dict(bn_layers)
            # bn_weights_dict = filter_conv(to_dict(bn_weights), bn_model)
            # bn_weights = map(lambda weight: weight.name,
            #                  filter(lambda weight: len(weight.shape) == 4, bn_model.trainable_weights))

            # naive_layers = [layer.name for layer in naive_model.layers]
            # naive_weights = [weight.name for weight in naive_model.trainable_weights]
            # naive_layers_dict = to_dict(naive_layers)
            # naive_weights_dict = filter_conv(to_dict(naive_weights), naive_model)
            # naive_weights = map(lambda weight: weight.name,
            #                     filter(lambda weight: len(weight.shape) == 4, naive_model.trainable_weights))

            layer_choices = LayerChoices()
            layer_choices.add_choices('ghd', ghd_layers)
            layer_choices.add_choices('ghd_weights', ghd_weights)
            # layer_choices.add_choices('bn', bn_layers)
            # layer_choices.add_choices('bn_weights', bn_weights)
            # layer_choices.add_choices('naive', naive_layers)
            # layer_choices.add_choices('naive_weights', naive_weights)

            realtime_model = RealtimeModel(graph, sess, layer_choices, learning_rate, batch_size, train_gen, val_gen,
                                           test_gen,
                                           logdir='tensorboard/%s' % get_current_time_in_string())
            realtime_model.add_model('ghd', ghd_model, ghd_layers_dict, ghd_weights_dict)
            # realtime_model.add_model('bn', bn_model, bn_layers_dict, bn_weights_dict)
            # realtime_model.add_model('naive', naive_model, naive_layers_dict, naive_weights_dict)

            realtime_model.init_tensorboard()

            thread = threading.Thread(target=visualise_thread, args=(img_input, realtime_model))
            thread.setDaemon(True)
            thread.start()

            # realtime_model.train_on_epoch()
            realtime_model.setting_window.mainloop()

    print('closing -> %s' % sess.close())


if __name__ == '__main__':
    main()
