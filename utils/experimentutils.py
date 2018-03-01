import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.models import Sequential

import imageutils
import visutils
from gui.window import SettingWindow


class SummaryCallback(Callback):
    def __init__(self, writer, summaries, global_steps, test_gen, interval):
        super(SummaryCallback, self).__init__()
        self.writer = writer  # type: tf.summary.FileWriter
        self.summaries = summaries
        self.global_steps = global_steps
        self.test_gen = test_gen
        self.interval = interval

    def add_summary(self, steps):
        # pass
        f = K.function([self.model.layers[0].input] + self.model.targets + self.model.sample_weights,
                       [self.summaries])
        x, y = next(self.test_gen)
        summ = f([x, y, [None]])[0]
        self.writer.add_summary(summ, global_step=steps)

    def on_train_begin(self, logs=None):
        super(SummaryCallback, self).on_train_begin(logs)
        if self.global_steps == 1:
            self.add_summary(0)

    def on_batch_end(self, batch, logs=None):
        if batch % self.interval == 0:
            self.add_summary(self.global_steps)
            self.global_steps += 1

    # def on_epoch_end(self, epoch, logs=None):
    #     super(SummaryCallback, self).on_epoch_end(epoch, logs)
    #     self.add_summary(self.global_steps)
    #     self.global_steps += 1


class AbstractRealtimeModel(object):
    def button_onclick(self, button_type):
        pass

    def dropdrop_callback(self, model_name, selected_layer_name):
        pass

    def entry_callback(self, entry_type, value):
        pass


class RealtimeModel(AbstractRealtimeModel):
    def __init__(self, graph, sess, layer_choices, init_learning_rate, init_batch_size, train_gen, val_gen, test_gen,
                 logdir):
        self.graph = graph
        self.sess = sess
        self.logdir = logdir
        self.learning_rate = init_learning_rate
        self.batch_size = init_batch_size
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.val_gen = val_gen
        self.no_epoch = 2
        self.models = {}
        self.models_layers_dict = {}
        self.models_weights_dict = {}
        self.models_selected_layer_name = {}
        self.models_selected_weight_name = {}
        self.button_running = False
        self.setting_window = SettingWindow(layer_choices, self.button_onclick, self.dropdown_callback,
                                            self.entry_callback)
        self.global_steps = 1
        self.is_changed = True
        self.activation_per_filter = True
        self.weight_per_filter = True
        self.activation_heatmap = True
        self.weight_heatmap = True

    def add_model(self, model_name, model, layers_dict, weights_dict):
        self.models[model_name] = model
        self.models_layers_dict[model_name] = layers_dict
        self.models_weights_dict[model_name + '_weights'] = weights_dict
        self.models_selected_layer_name[model_name] = layers_dict.keys()[0]
        self.models_selected_weight_name[model_name + '_weights'] = weights_dict.keys()[0]

    def button_onclick(self, button_type):
        self.setting_window.execute_entry_callback()
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
                                epochs=self.no_epoch,
                                validation_data=self.val_gen,
                                initial_epoch=0,
                                callbacks=[SummaryCallback(self.writer,
                                                           self.summaries[model_name],
                                                           self.global_steps,
                                                           self.test_gen,
                                                           10)])

        self.global_steps += int(self.no_epoch * (self.train_gen.batch_size / self.train_gen.n))  # epoch
        # self.global_steps += self.train_gen.n / self.batch_size # batch

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

    def dropdown_callback(self, dropdown_name, selected_layer_name):
        if dropdown_name in self.models_layers_dict and selected_layer_name in self.models_layers_dict[dropdown_name]:
            self.models_selected_layer_name[dropdown_name] = selected_layer_name
        else:
            self.models_selected_weight_name[dropdown_name] = selected_layer_name

        self.is_changed = True

    def entry_callback(self, entry_type, value):
        if entry_type == 'no_epoch':
            self.no_epoch = int(value)

    def get_weight(self, model, weight_idx):
        print('weight idx -> %s, name -> %s' % (weight_idx, model.trainable_weights[weight_idx].name))
        print('weight per filter -> %s, heatmap -> %s' % (self.weight_per_filter, self.weight_heatmap))
        weight = model.trainable_weights[weight_idx]
        weight = K.batch_get_value([weight])[0]
        first_layer = weight.shape[2] == 1 or weight.shape[2] == 3
        print('weight.shape -> %s' % list(weight.shape))
        weight = imageutils.normalize_weights(weight, 'conv', self.weight_per_filter, self.weight_heatmap, first_layer)
        perm = (3, 0, 1, 2) if first_layer else (3, 0, 1, 2, 4)
        weight = np.transpose(weight, perm)  # h,w,c,n -> n,h,w,c
        disp_w = 200 if first_layer else 800
        gap = 2 if first_layer else 5
        weight_disp = imageutils.combine_and_fit(weight, is_weights=True, disp_w=disp_w, gap=gap)
        return weight_disp

    def get_activation(self, img, model, model_layer_idx):
        out = visutils.activation(img, model, model_layer_idx)

        if len(out.shape) == 4:
            out = np.transpose(out, (3, 1, 2, 0))
        else:
            out = np.transpose(out, (1, 0))

        print('activation per filter -> %s, heatmap -> %s' % (self.activation_per_filter, self.activation_heatmap))
        out = imageutils.normalize(out, self.activation_per_filter, self.activation_heatmap)
        disp = imageutils.combine_and_fit(out, disp_w=300)
        return disp

    def visualise(self, img):
        for model_name, model in self.models.iteritems():
            print('------- %s -------' % model_name)
            layer_name = self.models_selected_layer_name[model_name]
            layer_idx = self.models_layers_dict[model_name][layer_name]

            weight_name = self.models_selected_weight_name[model_name + '_weights']
            weight_idx = self.models_weights_dict[model_name + '_weights'][weight_name]

            disp = self.get_activation(img, model, layer_idx)
            weight_disp = self.get_weight(model, weight_idx)

            cv2.imshow(model_name, disp)
            cv2.imshow('weight_' + model_name, weight_disp)

            res = model.predict(img)[0]
            res = self.parse_predict_result(res)
            print('prediction result -> %s' % str(res))
            print
        print

    def parse_predict_result(self, res):
        pred = np.argmax(res)
        prob = res[pred]
        return pred, prob

    def record_in_tensorboard(self, summary_inputs, postfix_name, tensor):
        min_weight = K.min(tensor)
        max_weight = K.max(tensor)
        mean_weight = K.mean(tensor)

        if postfix_name != '':
            postfix_name = '_' + postfix_name

        summary_inputs.append(tf.summary.scalar('min' + postfix_name, min_weight))
        summary_inputs.append(tf.summary.scalar('max' + postfix_name, max_weight))
        summary_inputs.append(tf.summary.scalar('mean' + postfix_name, mean_weight))

        flatten_weight = K.flatten(tensor)
        summary_inputs.append(tf.summary.histogram('histogram' + postfix_name, flatten_weight))

    def init_tensorboard(self):
        self.summaries = {}
        self.writer = tf.summary.FileWriter(logdir=self.logdir)
        print('tensorboard logdir -> %s' % self.logdir)

        # min, max, mean, histogram for weights & activations, loss
        for model_name, model in self.models.iteritems():  # type: Sequential
            with tf.name_scope('%s_statistics' % model_name):
                summary_inputs = []
                for i, layer in enumerate(model.layers):
                    with tf.name_scope('layer_%s' % layer.name):
                        for weight in layer.trainable_weights:
                            weight_name = weight.name.split('/')[-1]
                            weight_name = weight_name[:-2] + '_' + weight_name[-1]  # xxx:0 -> xxx_0
                            weight_namescope = 'weight_%s' % weight_name
                            with tf.name_scope(weight_namescope):
                                self.record_in_tensorboard(summary_inputs, '', weight)

                        if i == 0:
                            self.record_in_tensorboard(summary_inputs, 'input', layer.input)

                        self.record_in_tensorboard(summary_inputs, 'layer', layer.output)

                summary_inputs.append(tf.summary.scalar('total_loss', model.total_loss))

                self.summaries[model_name] = tf.summary.merge(summary_inputs)
