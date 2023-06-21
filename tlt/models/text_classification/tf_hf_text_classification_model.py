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
import tensorflow as tf

from transformers import BertTokenizer

from downloader.models import ModelDownloader
from tlt import TLT_BASE_DIR
from tlt.models.text_classification.tf_text_classification_model import TFTextClassificationModel
from tlt.datasets.text_classification.text_classification_dataset import TextClassificationDataset
from tlt.utils.dataset_utils import prepare_huggingface_input_data
from tlt.utils.file_utils import read_json_file


class TFHFTextClassificationModel(TFTextClassificationModel):
    """
    Class to represent a TensorFlow pretrained model from Hugging Face that can be used for binary text classification
    fine tuning.
    """

    def __init__(self, model_name: str, model=None, **kwargs):
        tfhub_model_map = read_json_file(os.path.join(
            TLT_BASE_DIR, "models/configs/tf_hf_text_classification_models.json"))
        if model_name not in tfhub_model_map.keys():
            raise ValueError("The specified TF hugging face text classification model ({}) "
                             "is not supported.".format(model_name))

        self._max_seq_length = kwargs["max_seq_length"] if "max_seq_length" in kwargs else 128
        self._hub_name = tfhub_model_map[model_name]["hub_name"]

        # extra properties that should become configurable in the future
        self._dropout_layer_rate = 0.1
        self._epsilon = 1e-08
        self._generate_checkpoints = True

        TFTextClassificationModel.__init__(self, model_name, model, **kwargs)

    @property
    def hub_name(self):
        """ Name of the model in Hugging Face """
        return self._hub_name

    @property
    def num_classes(self):
        return self._num_classes

    def _get_hub_model(self, num_classes, extra_layers=None):
        if not self._model:
            tf_bert_downloader = ModelDownloader(self._hub_name, hub='tf_bert_huggingface')
            bert_model = tf_bert_downloader.download()

            input_ids = tf.keras.layers.Input(shape=(self._max_seq_length,), dtype=tf.int32, name="input_ids")
            attention_mask = tf.keras.layers.Input(shape=(self._max_seq_length,), dtype=tf.int32, name='attention_mask')
            bert_output = bert_model.bert(input_ids, attention_mask=attention_mask)[1]

            if extra_layers:
                for layer_size in extra_layers:
                    bert_output = tf.keras.layers.Dense(layer_size, "relu")(bert_output)

            if self._dropout_layer_rate is not None:
                bert_output = tf.keras.layers.Dropout(self._dropout_layer_rate)(bert_output)

            dense_layer_dims = num_classes

            # For binary classification we only need 1 dimension
            if num_classes == 2:
                dense_layer_dims = 1

            output = tf.keras.layers.Dense(dense_layer_dims, activation=None, name='classifier')(bert_output)
            self._model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

            self._model.summary(print_fn=print)

        self._num_classes = num_classes
        return self._model

    def train(self, dataset: TextClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              do_eval=True, early_stopping=False, lr_decay=True, enable_auto_mixed_precision=None,
              shuffle_files=True, extra_layers=None, seed=None, distributed=False, hostfile=None, nnodes=1,
              nproc_per_node=1, **kwargs):
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
               do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                    at the end of each epoch.
               early_stopping (bool): Enable early stopping if convergence is reached while training
                    at the end of each epoch.
               lr_decay (bool): If lr_decay is True and do_eval is True, learning rate decay on the validation loss
                    is applied at the end of each epoch.
               enable_auto_mixed_precision (bool or None): Enable auto mixed precision for training. Mixed precision
                    uses both 16-bit and 32-bit floating point types to make training run faster and use less memory.
                    It is recommended to enable auto mixed precision training when running on platforms that support
                    bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                    does not support bfloat16, it can be detrimental to the training performance. If
                    enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                    running with Intel fourth generation Xeon processors, and disabled for other platforms.
               shuffle_files (bool): Boolean specifying whether to shuffle the training data before each epoch.
               extra_layers (list[int]): Optionally insert additional dense layers between the base model and output
                    layer. This can help increase accuracy when fine-tuning a TFHub model. The input should be a list of
                    integers representing the number and size of the layers, for example [1024, 512] will insert two
                    dense layers, the first with 1024 neurons and the second with 512 neurons.
               seed (int): Optionally set a seed for reproducibility.

           Returns:
               History object from the model.fit() call

           Raises:
               FileExistsError: if the output directory is a file
               TypeError: if the dataset specified is not a TextClassificationDataset
               TypeError: if the output_dir parameter is not a string
               TypeError: if the epochs parameter is not a integer
               TypeError: if the initial_checkpoints parameter is not a string
               TypeError: if the extra_layers parameter is not a list of integers
        """
        self._check_train_inputs(output_dir, dataset, TextClassificationDataset, epochs, initial_checkpoints)

        if extra_layers:
            if not isinstance(extra_layers, list):
                raise TypeError("The extra_layers parameter must be a list of ints but found {}".format(
                    type(extra_layers)))
            else:
                for layer in extra_layers:
                    if not isinstance(layer, int):
                        raise TypeError("extra_layers must be a list of ints but found a list containing {}".format(
                            type(layer)))

        dataset_num_classes = len(dataset.class_names)

        self._set_seed(seed)

        # Set auto mixed precision
        self.set_auto_mixed_precision(enable_auto_mixed_precision)

        # If the number of classes doesn't match what was used before, clear out the previous model
        if dataset_num_classes != self.num_classes:
            self._model = None

        self._model = self._get_hub_model(dataset_num_classes, extra_layers)

        callbacks, train_data, val_data = self._get_train_callbacks(dataset, output_dir, initial_checkpoints, do_eval,
                                                                    early_stopping, lr_decay, dataset_num_classes)

        if distributed:
            try:
                self._history = None
                saved_objects_dir = self.export_for_distributed(
                    export_dir=os.path.join(output_dir, "tlt_saved_objects"),
                    train_data=train_data,
                    val_data=val_data
                )
                self._fit_distributed(saved_objects_dir, epochs, shuffle_files, hostfile, nnodes, nproc_per_node,
                                      kwargs.get('use_horovod'), self._hub_name, self._max_seq_length)
            finally:
                self.cleanup_saved_objects_for_distributed()
        else:
            tokenized_data, labels = prepare_huggingface_input_data(train_data, self._hub_name, self._max_seq_length)

            if val_data:
                tokenized_val_data, val_labels = prepare_huggingface_input_data(val_data, self._hub_name,
                                                                                self._max_seq_length)
                val_data = ([tokenized_val_data['input_ids'], tokenized_val_data['attention_mask']],
                            tf.convert_to_tensor(val_labels))

            batch_size = dataset._preprocessed["batch_size"] if dataset._preprocessed and \
                "batch_size" in dataset._preprocessed else None

            history = self._model.fit(
                [tokenized_data['input_ids'], tokenized_data['attention_mask']],
                tf.convert_to_tensor(labels), batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                validation_data=val_data)

            self._history = history.history

        return self._history

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
               TypeError: if the dataset specified is not a TextClassificationDataset
               ValueError: if the use_test_set=True and no test subset has been defined in the dataset.
               ValueError: if the model has not been trained or loaded yet.
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

        tokenized_data, labels = prepare_huggingface_input_data(eval_dataset, self._hub_name, self._max_seq_length)

        return self._model.evaluate([tokenized_data['input_ids'], tokenized_data['attention_mask']],
                                    tf.convert_to_tensor(labels))

    def predict(self, input_samples):
        """
           Generates predictions for the specified input samples.

           Args:
               input_samples (str, list, numpy array, tensor, tf.data dataset or a generator keras.utils.Sequence):
                    Input samples to use to predict. These will be sent to the tf.keras.Model predict() function.

           Returns:
               Numpy array of scores

           Raises:
               ValueError: if the model has not been trained or loaded yet.
               ValueError: if there is a mismatch between the input_samples and the model's expected input.
        """
        if self._model is None:
            raise ValueError("The model must be trained or loaded before predicting.")

        # If a single string is passed in, make it a list so that it's compatible with the keras model predict
        if isinstance(input_samples, str):
            input_samples = [input_samples]

        tokenizer = BertTokenizer.from_pretrained(self._hub_name)

        if tf.is_tensor(input_samples):
            converted_batch = []
            for x in input_samples:
                converted_batch.append(bytes.decode(x.numpy()))
            input_samples = converted_batch

        encoded_input = dict(tokenizer(input_samples, padding='max_length', truncation=True,
                                       max_length=self._max_seq_length, return_tensors="tf"))

        return tf.sigmoid(self._model.predict(encoded_input)).numpy()
