#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
# SPDX-License-Identifier: EPL-2.0
#

import os
import tensorflow as tf
import tensorflow_hub as hub

from tlk.models.tfhub_model import TFHubModel
from tlk.models.image_classification.image_classification_model import ImageClassificationModel
from tlk.datasets.image_classification.image_classification_dataset import ImageClassificationDataset
from tlk.utils.types import FrameworkType, UseCaseType

# Dictionary of TFHub image classification models.
# TODO: Probably move this to json and standardize it with other models
tfhub_model_map = {
    "resnet_v1_50": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/5",
        "image_size": 224
    },
    "resnet_v2_50": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5",
        "image_size": 224
    },
    "resnet_v2_101": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/resnet_v2_101/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/5",
        "image_size": 224
    },
    "mobilenet_v2_100_224": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
        "image_size": 224
    },
    "efficientnetv2-s": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2",
        "feature_vector": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
        "image_size": 384
    },
    "efficientnet_b0": {
        "imagenet_model": "https://tfhub.dev/google/efficientnet/b0/classification/1",
        "feature_vector": "https://tfhub.dev/google/efficientnet/b0/feature-vector/1",
        "image_size": 224
    },
    "efficientnet_b1": {
        "imagenet_model": "https://tfhub.dev/google/efficientnet/b1/classification/1",
        "feature_vector": "https://tfhub.dev/google/efficientnet/b1/feature-vector/1",
        "image_size": 240
    },
    "efficientnet_b2": {
        "imagenet_model": "https://tfhub.dev/google/efficientnet/b2/classification/1",
        "feature_vector": "https://tfhub.dev/google/efficientnet/b2/feature-vector/1",
        "image_size": 260
    },
    "inception_v3": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/inception_v3/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",
        "image_size": 299
    },
    "nasnet_large": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/nasnet_large/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/5",
        "image_size": 331
    }
}


class TFHubImageClassificationModel(ImageClassificationModel, TFHubModel):
    """
    Class used to represent a TF Hub pretrained model
    """

    def __init__(self, model_name: str):
        if model_name not in tfhub_model_map.keys():
            raise ValueError("The specified TF Hub image classification model ({}) "
                             "is not supported.".format(model_name))

        self._model_url = tfhub_model_map[model_name]["imagenet_model"]
        self._feature_vector_url = tfhub_model_map[model_name]["feature_vector"]
        self._image_size = tfhub_model_map[model_name]["image_size"]

        # extra properties that will become configurable in the future
        self._do_fine_tuning = False
        self._dropout_layer_rate = None
        self._optimizer = tf.keras.optimizers.Adam()
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # placeholder for model definition
        self._model = None
        self._num_classes = None

        TFHubModel.__init__(self, self._model_url, model_name, FrameworkType.TENSORFLOW,
                            UseCaseType.IMAGE_CLASSIFICATION)
        ImageClassificationModel.__init__(self, self._image_size, self._do_fine_tuning, self._dropout_layer_rate,
                                          self._model_name, self._framework, self._use_case)

    @property
    def feature_vector_url(self):
        return self._feature_vector_url

    @property
    def num_classes(self):
        return self._num_classes

    def _get_hub_model(self, num_classes):
        if not self._model:
            feature_extractor_layer = hub.KerasLayer(self.feature_vector_url,
                                                     input_shape=(self.image_size, self.image_size, 3),
                                                     trainable=self.do_fine_tuning)

            if self.dropout_layer_rate is None:
                self._model = tf.keras.Sequential([
                    feature_extractor_layer,
                    tf.keras.layers.Dense(num_classes)
                ])
            else:
                self._model = tf.keras.Sequential([
                    feature_extractor_layer,
                    tf.keras.layers.Dropout(self.dropout_layer_rate),
                    tf.keras.layers.Dense(num_classes)
                ])

            print(self._model.summary())

        self._num_classes = num_classes
        return self._model

    def load_from_directory(self,  model_dir: str):
        raise NotImplementedError("Loading the model from a directory has not been implemented yet")

    def train(self, dataset: ImageClassificationDataset, epochs=1):
        dataset_num_classes = len(dataset.class_names)

        # If the number of classes doesn't match what was used before, clear out the previous model
        if dataset_num_classes != self.num_classes:
            self._model = None

        self._model = self._get_hub_model(dataset_num_classes)

        self._model.compile(
            optimizer=self._optimizer,
            loss=self._loss,
            metrics=['acc'])

        class CollectBatchStats(tf.keras.callbacks.Callback):
            def __init__(self):
                self.batch_losses = []
                self.batch_acc = []

            def on_train_batch_end(self, batch, logs=None):
                self.batch_losses.append(logs['loss'])
                self.batch_acc.append(logs['acc'])
                self.model.reset_metrics()

        batch_stats_callback = CollectBatchStats()

        return self._model.fit(dataset.dataset, epochs=epochs, shuffle=True, callbacks=[batch_stats_callback])

    def evaluate(self, dataset: ImageClassificationDataset):
        return self._model.evaluate(dataset.dataset)

    def predict(self, dataset: ImageClassificationDataset):
        return NotImplementedError("Predict has not been implemented")

    def export(self, output_dir):
        if self._model:
            # Save the model in a format that can be served
            saved_model_dir = os.path.join(output_dir, self.model_name)
            if os.path.exists(saved_model_dir) and len(os.listdir(saved_model_dir)):
                saved_model_dir = os.path.join(saved_model_dir, "{}".format(len(os.listdir(saved_model_dir)) + 1))
            else:
                saved_model_dir = os.path.join(saved_model_dir, "1")

            self._model.save(saved_model_dir)
            print("Saved model directory:", saved_model_dir)

            # TODO: Also save off model info and configs so that we can reload it
        else:
            raise ValueError("Unable to export the model, because it hasn't been trained yet")
