import threading
from collections import OrderedDict

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K

import dataset
from gui.window import SettingWindow, LayerChoices
from nets.keras_models import *
from utils import imageutils, visutils


class MnistImageInput(object):
    def __init__(self):
        from dataset import get_mnist
        self.X_test = get_mnist()[2][0]
        self.img_idx = 0

    def get_image(self):
        img_disp = self.X_test[self.img_idx]
        img_disp = cv2.resize(img_disp, (28, 28))
        img = img_disp[np.newaxis, :, :, np.newaxis]
        img = img.astype(np.float32)
        return img_disp, img

    def get_length(self):
        return len(self.X_test)

    def has_next_image(self):
        return self.img_idx > 0 or self.img_idx < self.get_length() - 1


class RealtimeModel(object):
    def __init__(self, graph, sess, layer_choices, init_learning_rate, init_batch_size, train_gen, val_gen):
        self.graph = graph
        self.sess = sess
        self.is_changed = True
        self.learning_rate = init_learning_rate
        self.batch_size = init_batch_size
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.initial_epoch = 0
        self.models = {}
        self.models_layers_dict = {}
        self.models_selected_layer_name = {}
        self.button_running = False
        self.setting_window = SettingWindow(layer_choices, self.button_onclick, self.dropdown_callback,
                                            self.entry_callback)

    def add_model(self, name, model, layers_dict):
        self.models[name] = model
        self.models_layers_dict[name] = layers_dict
        self.models_selected_layer_name[name] = layers_dict.keys()[0]

    def button_onclick(self, button_type):
        if not self.button_running:
            print('button running now')
            self.button_running = True
            if button_type == 'epoch':
                self.train_on_epoch()
            elif button_type == 'batch':
                self.train_on_batch()
            elif button_type == 'reset':
                self.reset()
            self.button_running = False
            self.is_changed = True
            print('button stopped')
        else:
            print('button running, please wait')

    def train_on_batch(self):
        print('train on batch')
        x, y = self.train_gen.next()
        for model_name, model in self.models.iteritems():  # type: Sequential
            model.train_on_batch(x, y)

    def train_on_epoch(self):
        print('train on epoch')
        for model_name, model in self.models.iteritems():  # type: Sequential
            model.fit_generator(self.train_gen,
                                steps_per_epoch=self.batch_size,
                                epochs=1,
                                validation_data=self.val_gen,
                                validation_steps=self.batch_size,
                                initial_epoch=0)

        self.initial_epoch += 1

    def reset(self):
        session = K.get_session()
        for model_name, model in self.models.iteritems():
            for layer in model.layers:
                for v in layer.__dict__:
                    v_arg = getattr(layer, v)
                    if hasattr(v_arg, 'initializer'):
                        initializer_method = getattr(v_arg, 'initializer')
                        initializer_method.run(session=session)
                        print('reinitializing layer {}.{}'.format(layer.name, v))

    def dropdown_callback(self, model_name, selected_layer_name):
        self.models_selected_layer_name[model_name] = selected_layer_name
        self.is_changed = True

    def entry_callback(self, entry_type, value):
        # if entry_type == 'learning_rate':
        #     self.learning_rate = float(value)
        # elif entry_type == 'batch_size':
        #     self.batch_size = int(value)

        # notes: not working yet
        pass

    def get_weight(self, model):
        weight = model.get_weights()[0]
        weight = imageutils.normalize_weights(weight, 'conv', heatmap=False)
        weight = np.transpose(weight, (3, 0, 1, 2))
        weight_disp = imageutils.combine_and_fit(weight, is_weights=True, disp_w=200)
        return weight_disp

    def get_activation(self, img, model, model_layer_idx):
        out = visutils.activation(img, model, model_layer_idx)

        if len(out.shape) == 4:
            is_conv = True
            is_fc = False
            out = np.transpose(out, (3, 1, 2, 0))
        else:
            is_conv = False
            is_fc = True
            out = np.transpose(out, (1, 0))

        out = imageutils.normalize(out)
        disp = imageutils.combine_and_fit(out, is_conv=is_conv, is_fc=is_fc, disp_w=300)
        return disp

    def visualise(self, img):
        for model_name, model in self.models.iteritems():
            layer_name = self.models_selected_layer_name[model_name]
            layer_idx = self.models_layers_dict[model_name][layer_name]

            disp = self.get_activation(img, model, layer_idx)
            weight_disp = self.get_weight(model)

            cv2.imshow(model_name, disp)
            cv2.imshow('weight_' + model_name, weight_disp)

            res = model.predict(img)[0]
            res = self.parse_predict_result(res)
            print('result -> %s' % str(res))

    def parse_predict_result(self, res):
        pred = np.argmax(res)
        prob = res[pred]
        return pred, prob

def to_dict(layers):
    ret = OrderedDict()
    for i, layer_name in enumerate(layers):
        ret[layer_name] = i
    return ret


def visualise_thread(img_input, realtime_model):
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


def main():
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            batch_size = 64
            learning_rate = 0.1

            ghd_model = ghd_mnist_model(learning_rate, True)
            # if os.path.exists('./keras_mnist_ghd.h5'):
            #     ghd_model.load_weights('./keras_mnist_ghd.h5')

            bn_model = bn_mnist_model(learning_rate)
            # if os.path.exists('./keras_mnist_bn.h5'):
            #     bn_model.load_weights('./keras_mnist_bn.h5')

            naive_model = naive_mnist_model(learning_rate / 100.)
            # if os.path.exists('./keras_mnist_naive.h5'):
            #     naive_model.load_weights('./keras_mnist_naive.h5')

            train_gen, val_gen, test_gen = dataset.get_mnist_generator(batch_size,
                                                                       zero_mean=False,
                                                                       unit_variance=False,
                                                                       horizontal_flip=True,
                                                                       rotation_range=10,
                                                                       width_shift_range=0.1)

            img_input = MnistImageInput()

            ghd_layers = [layer.name for layer in ghd_model.layers]
            ghd_layers_dict = to_dict(ghd_layers)
            bn_layers = [layer.name for layer in bn_model.layers]
            bn_layers_dict = to_dict(bn_layers)
            naive_layers = [layer.name for layer in naive_model.layers]
            naive_layers_dict = to_dict(naive_layers)

            layer_choices = LayerChoices()
            layer_choices.add_choices('ghd', ghd_layers)
            layer_choices.add_choices('bn', bn_layers)
            layer_choices.add_choices('naive', naive_layers)

            realtime_model = RealtimeModel(graph, sess, layer_choices, learning_rate, batch_size, train_gen, val_gen)
            realtime_model.add_model('ghd', ghd_model, ghd_layers_dict)
            realtime_model.add_model('bn', bn_model, bn_layers_dict)
            realtime_model.add_model('naive', naive_model, naive_layers_dict)

            thread = threading.Thread(target=visualise_thread, args=(img_input, realtime_model))
            thread.start()

            realtime_model.setting_window.mainloop()


if __name__ == '__main__':
    main()
