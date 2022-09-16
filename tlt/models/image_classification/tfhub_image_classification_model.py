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
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tlt import TLT_BASE_DIR
from tlt.models.image_classification.tf_image_classification_model import TFImageClassificationModel
from tlt.datasets.image_classification.image_classification_dataset import ImageClassificationDataset
from tlt.utils.file_utils import read_json_file, verify_directory
from tlt.utils.types import FrameworkType, UseCaseType


class TFHubImageClassificationModel(TFImageClassificationModel):
    """
    Class used to represent a TF Hub pretrained model
    """

    def __init__(self, model_name: str):
        """
        Class constructor
        """
        tfhub_model_map = read_json_file(os.path.join(
            TLT_BASE_DIR, "models/configs/tfhub_image_classification_models.json"))
        if model_name not in tfhub_model_map.keys():
            raise ValueError("The specified TF Hub image classification model ({}) "
                             "is not supported.".format(model_name))

        self._model_url = tfhub_model_map[model_name]["imagenet_model"]
        self._feature_vector_url = tfhub_model_map[model_name]["feature_vector"]

        TFImageClassificationModel.__init__(self, model_name=model_name, model=None)

        # placeholder for model definition
        self._model = None
        self._num_classes = None
        self._image_size = tfhub_model_map[model_name]["image_size"]


    @property
    def model_url(self):
        """
        The public URL used to download the TFHub model
        """
        return self._model_url


    @property
    def feature_vector_url(self):
        """
        The public URL used to download the headless TFHub model used for transfer learning
        """
        return self._feature_vector_url

    def _get_hub_model(self, num_classes, extra_layers=None):
        if not self._model:
            feature_extractor_layer = hub.KerasLayer(self.feature_vector_url,
                                                     input_shape=(self.image_size, self.image_size, 3),
                                                     trainable=self.do_fine_tuning)

            self._model = tf.keras.Sequential([feature_extractor_layer])

            if extra_layers:
                for layer_size in extra_layers:
                    self._model.add(tf.keras.layers.Dense(layer_size, "relu"))

            if self.dropout_layer_rate is not None:
                self._model.add(tf.keras.layers.Dropout(dropout_layer_rate))

            self._model.add(tf.keras.layers.Dense(num_classes))

            self._model.summary(print_fn=print)

        self._num_classes = num_classes
        return self._model

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              do_eval=True, enable_auto_mixed_precision=None, shuffle_files=True, seed=None, extra_layers=None):
        """ 
            Trains the model using the specified image classification dataset. The first time training is called, it
            will get the feature extractor layer from TF Hub and add on a dense layer based on the number of classes
            in the specified dataset. The model is compiled and trained for the specified number of epochs. If a
            path to initial checkpoints is provided, those weights are loaded before training.

            Args:
                dataset (ImageClassificationDataset): Dataset to use when training the model
                output_dir (str): Path to a writeable directory for checkpoint files
                epochs (int): Number of epochs to train the model (default: 1)
                initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                    latest checkpoint will be used.
                do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                    at the end of each epoch.
                enable_auto_mixed_precision (bool or None): Enable auto mixed precision for training. Mixed precision
                    uses both 16-bit and 32-bit floating point types to make training run faster and use less memory.
                    It is recommended to enable auto mixed precision training when running on platforms that support
                    bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                    does not support bfloat16, it can be detrimental to the training performance. If
                    enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                    running with Intel fourth generation Xeon processors, and disabled for other platforms.
                shuffle_files (bool): Boolean specifying whether to shuffle the training data before each epoch.
                seed (int): Optionally set a seed for reproducibility.
                extra_layers (list[int]): Optionally insert additional dense layers between the base model and output
                    layer. This can help increase accuracy when fine-tuning a TFHub model. The input should be a list of
                    integers representing the number and size of the layers, for example [1024, 512] will insert two
                    dense layers, the first with 1024 neurons and the second with 512 neurons. 

            Returns:
                History object from the model.fit() call

            Raises:
               FileExistsError if the output directory is a file
               TypeError if the dataset specified is not an ImageClassificationDataset
               TypeError if the output_dir parameter is not a string
               TypeError if the epochs parameter is not a integer
               TypeError if the initial_checkpoints parameter is not a string
               TypeError if the extra_layers parameter is not a list of integers
        """

        verify_directory(output_dir)

        if not isinstance(dataset, ImageClassificationDataset):
            raise TypeError("The dataset must be a ImageClassificationDataset but found a {}".format(type(dataset)))

        if not isinstance(epochs, int):
            raise TypeError("Invalid type for the epochs arg. Expected an int but found a {}".format(type(epochs)))

        if initial_checkpoints and not isinstance(initial_checkpoints, str):
            raise TypeError("The initial_checkpoints parameter must be a string but found a {}".format(
                type(initial_checkpoints)))
        
        if extra_layers:
            if not isinstance(extra_layers, list):
                raise TypeError("The extra_layers parameter must be a list of ints but found {}".format(
                    type(extra_layers)))
            else:
                for layer in extra_layers:
                    if not isinstance(layer, int):
                        raise TypeError("The extra_layers parameter must be a list of ints but found a list containing {}".format(
                            type(layer)))

        dataset_num_classes = len(dataset.class_names)

        # If the number of classes doesn't match what was used before, clear out the previous model
        if dataset_num_classes != self.num_classes:
            self._model = None

        if seed is not None:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # Set auto mixed precision
        self.set_auto_mixed_precision(enable_auto_mixed_precision)

        self._model = self._get_hub_model(dataset_num_classes, extra_layers)

        self._model.compile(
            optimizer=self._optimizer,
            loss=self._loss,
            metrics=['acc'])

        if initial_checkpoints:
            if os.path.isdir(initial_checkpoints):
                initial_checkpoints = tf.train.latest_checkpoint(initial_checkpoints)

            self._model.load_weights(initial_checkpoints)

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
                filepath=os.path.join(checkpoint_dir, self.model_name.replace('/', '_')), save_weights_only=True)
            print("Checkpoint directory:", checkpoint_dir)
            callbacks.append(checkpoint_callback)

        if dataset._validation_type == 'shuffle_split':
            train_dataset =  dataset.train_subset
        else:
            train_dataset = dataset.dataset

        validation_data = dataset.validation_subset if do_eval else None

        return self._model.fit(train_dataset, epochs=epochs, shuffle=shuffle_files, callbacks=callbacks,
                               validation_data=validation_data)

    def evaluate(self, dataset: ImageClassificationDataset, use_test_set=False):
        """
        Evaluate the accuracy of the model on a dataset.

        If there is a validation subset, evaluation will be done on it (by default) or on the test set
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
        """
        Perform feed-forward inference and predict the classes of the input_samples
        """
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

