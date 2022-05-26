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
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tlk import TLK_BASE_DIR
from tlk.models.tfhub_model import TFHubModel
from tlk.models.image_classification.image_classification_model import ImageClassificationModel
from tlk.datasets.image_classification.image_classification_dataset import ImageClassificationDataset
from tlk.utils.file_utils import read_json_file, verify_directory
from tlk.utils.types import FrameworkType, UseCaseType


class TFHubImageClassificationModel(ImageClassificationModel, TFHubModel):
    """
    Class used to represent a TF Hub pretrained model
    """

    def __init__(self, model_name: str):
        tfhub_model_map = read_json_file(os.path.join(
            TLK_BASE_DIR, "models/configs/tfhub_image_classification_models.json"))
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
        self._generate_checkpoints = True

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

            self._model.summary(print_fn=print)

        self._num_classes = num_classes
        return self._model

    def load_from_directory(self,  model_dir: str):
        # Verify that the model directory exists
        verify_directory(model_dir, require_directory_exists=True)

        self._model = tf.keras.models.load_model(model_dir)
        self._model.summary(print_fn=print)

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1):
        verify_directory(output_dir)

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

        callbacks = [batch_stats_callback]

        # Create a callback for generating checkpoints
        if self._generate_checkpoints:
            checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(self.model_name))
            verify_directory(checkpoint_dir)
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, self.model_name), save_weights_only=True)
            print("Checkpoint directory:", checkpoint_dir)
            callbacks.append(checkpoint_callback)

        if dataset._validation_type == 'shuffle_split':
            train_dataset =  dataset.train_subset
        else:
            train_dataset = dataset.dataset

        return self._model.fit(train_dataset, epochs=epochs, shuffle=True, callbacks=callbacks)

    def evaluate(self, dataset: ImageClassificationDataset, use_test_set=False):
        """
        If there is a validation set, evaluation will be done on it (by default) or on the test set
        (by setting use_test_set=True). Otherwise, the entire non-partitioned dataset will be
        used for evaluation.
        """
        if use_test_set:
            if dataset.test_subset:
                eval_dataset = dataset.test_subset
            else:
                raise ValueError("No test subset is defined")
        elif dataset.validation_subset:
            eval_dataset = dataset.validation_subset
        else:
            eval_dataset = dataset.dataset

        if self._model is None:
            # The model hasn't been trained yet, use the original ImageNet trained model
            print("The model has not been trained yet, so evaluation is being done using the original model " + \
                  "and its classes")
            original_model = tf.keras.Sequential([
                hub.KerasLayer(self._model_url, input_shape=(self._image_size, self._image_size) + (3,))
            ])
            original_model.compile(
                optimizer=self._optimizer,
                loss=self._loss,
                metrics=['acc'])
            return original_model.evaluate(eval_dataset)
        else:
            return self._model.evaluate(eval_dataset)

    def predict(self, input_samples):
        if self._model is None:
            print("The model has not been trained yet, so predictions are being done using the original model")
            original_model = tf.keras.Sequential([
                hub.KerasLayer(self._model_url, input_shape=(self._image_size, self._image_size) + (3,))
            ])
            predictions = original_model.predict(input_samples)
        else:
            predictions = self._model.predict(input_samples)
        predicted_ids = np.argmax(predictions, axis=-1)
        return predicted_ids

    def export(self, output_dir):
        if self._model:
            # Save the model in a format that can be served
            verify_directory(output_dir)
            saved_model_dir = os.path.join(output_dir, self.model_name)
            if os.path.exists(saved_model_dir) and len(os.listdir(saved_model_dir)):
                saved_model_dir = os.path.join(saved_model_dir, "{}".format(len(os.listdir(saved_model_dir)) + 1))
            else:
                saved_model_dir = os.path.join(saved_model_dir, "1")

            self._model.save(saved_model_dir)
            print("Saved model directory:", saved_model_dir)

            return saved_model_dir
        else:
            raise ValueError("Unable to export the model, because it hasn't been trained yet")
