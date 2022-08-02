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

from tlt import TLT_BASE_DIR
from tlt.models.tfhub_model import TFHubModel
from tlt.models.text_classification.text_classification_model import TextClassificationModel
from tlt.datasets.text_classification.text_classification_dataset import TextClassificationDataset
from tlt.utils.file_utils import read_json_file, verify_directory
from tlt.utils.types import FrameworkType, UseCaseType

# Note that tensorflow_text isn't used directly but the import is required to register ops used by the
# BERT text preprocessor
import tensorflow_text


class TFHubTextClassificationModel(TextClassificationModel, TFHubModel):
    """
    Class used to represent a TF Hub pretrained model that can be used for binary text classification
    fine tuning.
    """

    def __init__(self, model_name: str):
        tfhub_model_map = read_json_file(os.path.join(
            TLT_BASE_DIR, "models/configs/tfhub_text_classification_models.json"))
        if model_name not in tfhub_model_map.keys():
            raise ValueError("The specified TF Hub text classification model ({}) "
                             "is not supported.".format(model_name))

        self._hub_preprocessor = tfhub_model_map[model_name]["preprocessor"]
        self._model_url = tfhub_model_map[model_name]["encoder"]
        self._checkpoint_zip = tfhub_model_map[model_name]["checkpoint_zip"]

        # extra properties that should become configurable in the future
        self._dropout_layer_rate = 0.1
        self._learning_rate = 3e-5
        self._epsilon = 1e-08
        self._generate_checkpoints = True

        # placeholder for model definition
        self._model = None
        self._num_classes = None

        TFHubModel.__init__(self, self._model_url, model_name, FrameworkType.TENSORFLOW,
                            UseCaseType.TEXT_CLASSIFICATION)
        TextClassificationModel.__init__(self, model_name, FrameworkType.TENSORFLOW, UseCaseType.TEXT_CLASSIFICATION,
                                         dropout_layer_rate=self._dropout_layer_rate)

    @property
    def preprocessor_url(self):
        return self._hub_preprocessor

    @property
    def num_classes(self):
        return self._num_classes

    def _get_hub_model(self, num_classes):
        if not self._model:
            input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='input_layer')
            preprocessing_layer = hub.KerasLayer(self._hub_preprocessor, name='preprocessing')
            encoder_inputs = preprocessing_layer(input_layer)
            encoder_layer = hub.KerasLayer(self._model_url, trainable=True, name='encoder')
            outputs = encoder_layer(encoder_inputs)
            net = outputs['pooled_output']

            if self._dropout_layer_rate is not None:
                net = tf.keras.layers.Dropout(self._dropout_layer_rate)(net)

            dense_layer_dims = num_classes

            # For binary classification we only need 1 dimension
            if num_classes == 2:
                dense_layer_dims = 1

            net = tf.keras.layers.Dense(dense_layer_dims, activation=None, name='classifier')(net)
            self._model = tf.keras.Model(input_layer, net)

            self._model.summary(print_fn=print)

        self._num_classes = num_classes
        return self._model

    def train(self, dataset: TextClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              enable_auto_mixed_precision=None, shuffle_files=True):
        """
           Trains the model using the specified binary text classification dataset. If a path to initial checkpoints is
           provided, those weights are loaded before training.
           
           Args:
               dataset (TextClassificationDataset): The dataset to use for training. If a train subset has been
                                                    defined, that subset will be used to fit the model. Otherwise, the
                                                    entire non-partitioned dataset will be used.
               output_dir (str): A writeable output directory to write checkpoint files during training
               epochs (int): The number of training epochs [default: 1]
               initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                    latest checkpoint will be used.
               enable_auto_mixed_precision (bool or None): Enable auto mixed precision for training. Mixed precision
                    uses both 16-bit and 32-bit floating point types to make training run faster and use less memory.
                    It is recommended to enable auto mixed precision training when running on platforms that support
                    bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                    does not support bfloat16, it can be detrimental to the training performance. If
                    enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                    running with Intel fourth generation Xeon processors, and disabled for other platforms.
               shuffle_files (bool): Boolean specifying whether to shuffle the training data before each epoch.

           Returns:
               History object from the model.fit() call

           Raises:
               FileExistsError if the output directory is a file
               TypeError if the dataset specified is not a TextClassificationDataset
               TypeError if the output_dir parameter is not a string
               TypeError if the epochs parameter is not a integer
               TypeError if the initial_checkpoints parameter is not a string
               NotImplementedError if the specified dataset has more than 2 classes.
        """
        verify_directory(output_dir)

        if not isinstance(dataset, TextClassificationDataset):
            raise TypeError("The dataset must be a TextClassificationDataset but found a {}".format(type(dataset)))

        if not isinstance(epochs, int):
            raise TypeError("Invalid type for the epochs arg. Expected an int but found a {}".format(type(epochs)))

        if initial_checkpoints and not isinstance(initial_checkpoints, str):
            raise TypeError("The initial_checkpoints parameter must be a string but found a {}".format(
                type(initial_checkpoints)))

        dataset_num_classes = len(dataset.class_names)

        if dataset_num_classes != 2:
            raise NotImplementedError("Training is only supported for binary text classification. The specified dataset"
                                      " has {} classes, but expected 2 classes.".format(dataset_num_classes))

        # Set auto mixed precision
        self.set_auto_mixed_precision(enable_auto_mixed_precision)

        # If the number of classes doesn't match what was used before, clear out the previous model
        if dataset_num_classes != self.num_classes:
            self._model = None

        self._model = self._get_hub_model(dataset_num_classes)
        print("Num dataset classes: ", dataset_num_classes)

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) if dataset_num_classes == 2 else \
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        metrics = tf.metrics.BinaryAccuracy() if dataset_num_classes == 2 else tf.keras.metrics.Accuracy()

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate, epsilon=self._epsilon),
            loss=loss,
            metrics=metrics)

        if initial_checkpoints:
            if os.path.isdir(initial_checkpoints):
                initial_checkpoints = tf.train.latest_checkpoint(initial_checkpoints)

            self._model.load_weights(initial_checkpoints)

        class CollectBatchStats(tf.keras.callbacks.Callback):
            def __init__(self):
                self.batch_losses = []
                self.batch_acc = []

            def on_train_batch_end(self, batch, logs=None):
                if logs and isinstance(logs, dict):

                    # Find the name of the accuracy key
                    accuracy_key = None
                    for log_key in logs.keys():
                        if 'acc' in log_key:
                            accuracy_key = log_key
                            break

                    self.batch_losses.append(logs['loss'])

                    if accuracy_key:
                        self.batch_acc.append(logs[accuracy_key])
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

        train_data = dataset.train_subset if dataset.train_subset else dataset.dataset
        val_data = dataset.validation_subset if dataset.validation_subset else None

        return self._model.fit(train_data, validation_data=val_data, epochs=epochs, shuffle=shuffle_files,
                               callbacks=callbacks)

    def evaluate(self, dataset: TextClassificationDataset, use_test_set=False):
        """
           If there is a validation set, evaluation will be done on it (by default) or on the test set (by setting 
           use_test_set=True). Otherwise, the entire non-partitioned dataset will be used for evaluation.
        
           Args:
               dataset (TextClassificationDataset): The dataset to use for evaluation. 
               use_test_set (bool): Specify if the test partition of the dataset should be used for evaluation.
                                    [default: False)

           Returns:
               Dictionary with loss and accuracy metrics

           Raises:
               TypeError if the dataset specified is not a TextClassificationDataset
               ValueError if the use_test_set=True and no test subset has been defined in the dataset.
               ValueError if the model has not been trained or loaded yet.
        """
        if not isinstance(dataset, TextClassificationDataset):
            raise TypeError("The dataset must be a TextClassificationDataset but found a {}".format(type(dataset)))

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
            raise ValueError("The model must be trained or loaded before evaluation.")

        return self._model.evaluate(eval_dataset)

    def predict(self, input_samples):
        """
           Generates predictions for the specified input samples.
        
           Args:
               input_samples (str, list, numpy array, tensor, tf.data dataset or a generator keras.utils.Sequence):
                    Input samples to use to predict. These will be sent to the tf.keras.Model predict() function.

           Returns:
               Numpy array of scores

           Raises:
               ValueError if the model has not been trained or loaded yet.
               ValueError if there is a mismatch between the input_samples and the model's expected input.
        """
        if self._model is None:
            raise ValueError("The model must be trained or loaded before predicting.")

        # If a single string is passed in, make it a list so that it's compatible with the keras model predict
        if isinstance(input_samples, str):
            input_samples = [input_samples]

        return tf.sigmoid(self._model.predict(input_samples)).numpy()
