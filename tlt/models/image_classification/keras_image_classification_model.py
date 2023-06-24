#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#

import os
from pydoc import locate
import tensorflow as tf

from downloader.models import ModelDownloader
from tlt import TLT_BASE_DIR
from tlt.models.image_classification.tf_image_classification_model import TFImageClassificationModel
from tlt.models.image_classification.tfhub_image_classification_model import TFHubImageClassificationModel
from tlt.utils.file_utils import read_json_file


class KerasImageClassificationModel(TFHubImageClassificationModel):
    """
    Class to represent a Keras.applications pretrained model for image classification
    """

    def __init__(self, model_name: str, model=None, optimizer=None, loss=None, **kwargs):
        """
        Class constructor
        """
        keras_model_map = read_json_file(os.path.join(
            TLT_BASE_DIR, "models/configs/tf_keras_image_classification_models.json"))
        if model_name not in keras_model_map.keys():
            raise ValueError("The specified Keras image classification model ({}) "
                             "is not supported.".format(model_name))

        TFImageClassificationModel.__init__(self, model_name=model_name, model=model, optimizer=optimizer,
                                            loss=loss, **kwargs)

        if self._model is None:
            self._num_classes = None
            self._image_size = keras_model_map[model_name]["image_size"]

        # Get the model-specific preprocessor from keras applications
        preprocessor_name = keras_model_map[model_name]["preprocessor"]
        if preprocessor_name is not None:
            self._preprocessor = locate('keras.applications.{}.preprocess_input'.format(preprocessor_name))

    def _model_downloader(self, model_name, include_top=False):
        downloader = ModelDownloader(model_name, hub='keras', model_dir=None, weights='imagenet',
                                     include_top=include_top)
        model = downloader.download()
        return model

    def _get_hub_model(self, num_classes, extra_layers=None):

        if not self._model:
            base_model = self._model_downloader(self._model_name)
            base_model.trainable = False
            inputs = tf.keras.Input(shape=(self._image_size, self._image_size, 3))

            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)

            if extra_layers:
                for layer_size in extra_layers:
                    x = tf.keras.layers.Dense(layer_size, activation='relu')(x)
            if self.dropout_layer_rate is not None:
                x = tf.keras.layers.Dropout(self.dropout_layer_rate)(x)
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            self._model = tf.keras.Model(inputs=inputs, outputs=outputs)

            self._model.summary(print_fn=print)

        self._num_classes = num_classes
        return self._model
