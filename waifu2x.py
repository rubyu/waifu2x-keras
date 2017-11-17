#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.advanced_activations import LeakyReLU
import json
from PIL import Image
from scipy import misc
import numpy as np


class Waifu2x:
    def __init__(self, model_path, block_size=128):
        self.block_size = block_size

        with open(model_path, 'rb') as f:
            self.params = json.load(f)

    def _load_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(
            filters=self.params[0]['nOutputPlane'],
            kernel_size=(self.params[0]['kH'], self.params[0]['kW']),
            strides=(1, 1),
            kernel_initializer='zeros',
            bias_initializer='zeros',
            padding='valid',
            weights=[np.array(self.params[0]['weight']).transpose(2, 3, 1, 0),
                     np.array(self.params[0]['bias'])],
            use_bias=True,
            input_shape=input_shape))
        model.add(LeakyReLU(0.1))
        for param in self.params[1:]:
            model.add(Conv2D(
                filters=param['nOutputPlane'],
                kernel_size=(param['kH'], param['kW']),
                strides=(1, 1),
                kernel_initializer='zeros',
                bias_initializer='zeros',
                padding='valid',
                weights=[np.array(param['weight']).transpose(2, 3, 1, 0),
                         np.array(param['bias'])],
                use_bias=True))
            if param != self.params[-1]:
                model.add(LeakyReLU(0.1))
        return model

    @property
    def input_channel(self):
        return self.params[0]['nInputPlane']

    @property
    def layers(self):
        return len(self.params)

    @staticmethod
    def _predict(model, x, num_layers, block_size):
        pad = num_layers
        x_bs = block_size
        y_bs = x_bs - num_layers * 2
        result = np.empty_like(x)
        max_i = int(math.ceil(result.shape[0] / y_bs)) - 1
        max_j = int(math.ceil(result.shape[1] / y_bs)) - 1
        x = np.pad(x,
                   ((pad, ((max_i + 1) * y_bs) - x.shape[0] + pad),
                    (pad, ((max_j + 1) * y_bs) - x.shape[1] + pad),
                    (0, 0)),
                   'edge')
        x_block = np.empty((1, x_bs, x_bs, x.shape[2]))
        for i in range(0, max_i + 1):
            for j in range(0, max_j + 1):
                x_block[0, 0:x_bs, 0:x_bs, :] = x[i * y_bs: i * y_bs + x_bs, j * y_bs: j * y_bs + x_bs, :]
                y = model.predict(x_block)
                y_h = y_w = y_bs
                if i == max_i:
                    y_h = result.shape[0] % y_bs
                if j == max_j:
                    y_w = result.shape[1] % y_bs
                result[i * y_bs: i * y_bs + y_h, j * y_bs: j * y_bs + y_w, :] = y[:, 0:y_h, 0:y_w, :]
        return result

    def _generate_1(self, im, scale):
        im = im.convert('YCbCr')
        if scale:
            im = im.resize((im.size[0] * scale, im.size[1] * scale), resample=Image.NEAREST)
        x = im.astype('float32') / 255
        model = self._load_model(input_shape=(self.block_size, self.block_size, 1))
        y = self._predict(model, x, num_layers=self.layers, block_size=self.block_size)
        im[:, :, 0] = np.clip(y, 0, 1) * 255
        return misc.toimage(im, mode='YCbCr').convert('RGB')

    def _generate_3(self, im, scale):
        if scale:
            im = im.resize((im.size[0] * scale, im.size[1] * scale), resample=Image.NEAREST)
        x = im.astype('float32') / 255
        model = self._load_model(input_shape=(self.block_size, self.block_size, 3))
        y = self._predict(model, x, num_layers=self.layers, block_size=self.block_size)
        im = np.clip(y, 0, 1) * 255
        return misc.toimage(im, mode='RGB')

    def generate(self, im, scale=None):
        """
        :param im: image file object
        :param scale: Scale factor. If 2 was given, each edge of `im` will be doubled before processing.
        :return: image file object
        """
        im = misc.fromimage(im)
        assert im.shape[-1] == self.input_channel

        if self.input_channel == 1:
            return self._generate_1(im, scale)
        else:
            return self._generate_3(im, scale)
