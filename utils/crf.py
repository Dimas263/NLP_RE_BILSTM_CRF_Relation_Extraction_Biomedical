# -*- coding: utf-8 -*-
from keras import backend as K
from keras.engine.topology import Layer


class CRF(Layer):
    def __init__(self, crf_size, **kwargs):
        self.crf_size = crf_size
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="W_{:s}".format(self.name), shape=(input_shape[-1], self.crf_size),initializer="glorot_normal",trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name), shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name), shape=(self.crf_size, 1), initializer="glorot_normal", trainable=True)
        super(CRF, self).build(input_shape)

    def call(self, x, mask=None):
        et = K.tanh(K.dot(x, self.W) + self.b)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


    def get_config(self):
        config = {"crf_size": self.crf_size}
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
