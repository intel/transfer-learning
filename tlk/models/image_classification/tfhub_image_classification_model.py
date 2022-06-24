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

import copy
import os
import numpy as np
import shutil
import tempfile
import tensorflow as tf
import tensorflow_hub as hub
import yaml

from tlk import TLK_BASE_DIR
from tlk.models.tfhub_model import TFHubModel
from tlk.models.image_classification.image_classification_model import ImageClassificationModel
from tlk.datasets.image_classification.image_classification_dataset import ImageClassificationDataset
from tlk.datasets.image_classification.tf_custom_image_classification_dataset import TFCustomImageClassificationDataset
from tlk.utils.file_utils import read_json_file, verify_directory
from tlk.utils.types import FrameworkType, UseCaseType
from tlk.utils.platform_util import PlatformUtil


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

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1, enable_auto_mixed_precision=None):
        """ 
            Trains the model using the specified image classification dataset. The first time training is called, it
            will get the feature extractor layer from TF Hub and add on a dense layer based on the number of classes
            in the specified dataset. The model is compiled and trained for the specified number of epochs.

            Args:
                dataset (ImageClassificationDataset): Dataset to use when training the model
                output_dir (str): Path to a writeable directory for checkpoint files
                epochs (int): Number of epochs to train the model (default: 1)
                enable_auto_mixed_precision (bool or None): Enable auto mixed precision for training. Mixed precision
                    uses both 16-bit and 32-bit floating point types to make training run faster and use less memory.
                    If enable_auto_mixed_precision is set to None, auto mixed precision will be enabled when running with
                    Intel fourth generation Xeon processors, and disabled for other platforms.

            Returns:
                History object from the model.fit() call
        """

        verify_directory(output_dir)

        dataset_num_classes = len(dataset.class_names)

        # If the number of classes doesn't match what was used before, clear out the previous model
        if dataset_num_classes != self.num_classes:
            self._model = None

        if enable_auto_mixed_precision is not None and not isinstance(enable_auto_mixed_precision, bool):
            raise TypeError("Invalid type for enable_auto_mixed_precision. Expected None or a bool.")

        # Get the TF version
        tf_major_version = 0
        tf_minor_version = 0
        if tf.version.VERSION is not None and '.' in tf.version.VERSION:
            tf_version_list = tf.version.VERSION.split('.')
            if len(tf_version_list) > 1:
                tf_major_version = int(tf_version_list[0])
                tf_minor_version = int(tf_version_list[1])

        auto_mixed_precision_supported = (tf_major_version == 2 and tf_minor_version >= 9) or tf_major_version > 2

        if enable_auto_mixed_precision is None:
            # Determine whether or not to enable this based on the CPU type
            try:
                # Only enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil(args=None).cpu_type == 'SPR'
            except Exception as e:
                if auto_mixed_precision_supported:
                    print("Unable to determine the CPU type:", str(e))
                enable_auto_mixed_precision = False
        elif not auto_mixed_precision_supported:
            print("Warning: Auto mixed precision requires TensorFlow 2.9.0 or later (found {}).".format(
                tf.version.VERSION))

        if auto_mixed_precision_supported:
            if enable_auto_mixed_precision:
                print("Enabling auto_mixed_precision_mkl")
            tf.config.optimizer.set_experimental_options({'auto_mixed_precision_mkl': enable_auto_mixed_precision})

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

    def write_inc_config_file(self, config_file_path, dataset, batch_size, overwrite=False,
                              resize_interpolation='bicubic', accuracy_criterion_relative=0.01, exit_policy_timeout=0,
                              exit_policy_max_trials=50, tuning_random_seed=9527,
                              tuning_workspace=''):
        """
        Writes an INC compatible config file to the specified path usings args from the specified dataset and
        parameters. This is currently only supported for TF custom image classification datasets.
        
        Args:
            config_file_path (str): Destination path on where to write the .yaml config file.
            dataset (BaseDataset): A tlk dataset object
            batch_size (int): Batch size to use for quantization and evaluation
            overwrite (bool): Specify whether or not to overwrite the config_file_path, if it already exists
                              (default: False)
            resize_interpolation (str): Interpolation type. Select from: 'bilinear', 'nearest', 'bicubic'
                                        (default: bicubic)
            accuracy_criterion_relative (float): Relative accuracy loss (default: 0.01, which is 1%)
            exit_policy_timeout (int): Tuning timeout in seconds (default: 0). Tuning processing finishes when the
                                       timeout or max_trials is reached. A tuning timeout of 0 means that the tuning
                                       phase stops when the accuracy criterion is met.
            exit_policy_max_trials (int): Maximum number of tuning trials (default: 50). Tuning processing finishes when
                                          the timeout or or max_trials is reached.
            tuning_random_seed (int): Random seed for deterministic tuning (default: 9527).
            tuning_workspace (dir): Path the INC nc_workspace folder. If the string is empty and the OUTPUT_DIR env var
                                    is set, that output directory will be used. If the string is empty and the
                                    OUTPUT_DIR env var is not set, the default INC nc_workspace location will be used.

        Returns:
            None

        Raises:
            FileExistsError if the config file already exists and overwrite is set to False.
            ValueError if the parameters are not within the expected values
            NotImplementedError if the dataset type is not TFCustomImageClassificationDataset.
        """
        if os.path.isfile(config_file_path) and not overwrite:
            raise FileExistsError('A file already exists at: {}. Provide a new file path or set overwrite=True',
                                  config_file_path)

        # We can setup the a custom dataset to use the ImageFolder dataset option in INC. They don't have a TFDS option,
        # so for now, we only support custom datasets for quantization
        if dataset is not TFCustomImageClassificationDataset and type(dataset) != TFCustomImageClassificationDataset:
            raise NotImplementedError('tlk quantization has only been implemented for TF image classification models '
                                      'with custom datasets')

        if batch_size and not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Invalid value for batch size ({}). Expected a positive integer.'.format(batch_size))

        if resize_interpolation not in ['bilinear', 'nearest', 'bicubic']:
            raise ValueError('Invalid value for resize interpolation ({}). Expected one of the following values: '
                             'bilinear, nearest, bicubic'.format(resize_interpolation))

        if accuracy_criterion_relative and not isinstance(accuracy_criterion_relative, float) or \
                not (0.0 <= accuracy_criterion_relative <= 1.0):
            raise ValueError('Invalid value for the accuracy criterion ({}). Expected a float value between 0.0 '
                             'and 1.0'.format(accuracy_criterion_relative))

        if exit_policy_timeout and not isinstance(exit_policy_timeout, int) or exit_policy_timeout < 0:
            raise ValueError('Invalid value for the exit policy timeout ({}). Expected a positive integer or 0.'.
                             format(exit_policy_timeout))

        if exit_policy_max_trials and not isinstance(exit_policy_max_trials, int) or exit_policy_max_trials < 1:
            raise ValueError('Invalid value for max trials ({}). Expected an integer greater than 0.'.
                             format(exit_policy_timeout))

        if tuning_random_seed and not isinstance(tuning_random_seed, int) or tuning_random_seed < 0:
            raise ValueError('Invalid value for tuning random seed ({}). Expected a positive integer.'.
                             format(tuning_random_seed))

        if not isinstance(tuning_workspace, str):
            raise ValueError('Invalid value for the nc_workspace directory. Expected a string.')

        # Get the image recognition INC template
        config_template = ImageClassificationModel.get_inc_config_template_dict(self)

        # Collect the different data loaders into a list, so that we can update them all the with the data transforms
        dataloader_configs = []

        # If tuning_workspace is undefined, use the OUTPUT_DIR, if the env var exists
        if not tuning_workspace:
            output_dir_env_var = os.getenv('OUTPUT_DIR', '')

            if output_dir_env_var:
                tuning_workspace = os.path.join(output_dir_env_var, 'nc_workspace')

        print("tuning_workspace:", tuning_workspace)

        if "quantization" in config_template.keys() and "calibration" in config_template["quantization"].keys() and \
            "dataloader" in config_template["quantization"]["calibration"].keys():
            dataloader_configs.append(config_template["quantization"]["calibration"]["dataloader"])

        if "evaluation" in config_template.keys():
            if "accuracy" in config_template["evaluation"].keys() and \
                            "dataloader" in config_template["evaluation"]["accuracy"].keys():
                dataloader_configs.append(config_template["evaluation"]["accuracy"]["dataloader"])
            if "performance" in config_template["evaluation"].keys() and \
                            "dataloader" in config_template["evaluation"]["performance"].keys():
                dataloader_configs.append(config_template["evaluation"]["performance"]["dataloader"])

        transform_config = {
            "PaddedCenterCrop": {
                "size": self.image_size,
                "crop_padding": 32
            },
            "Resize": {
                "size": self.image_size,
                "interpolation": resize_interpolation
            },
            "Rescale": {}
        }

        # Update the data loader configs
        for dataloader_config in dataloader_configs:
            # Set the transform configs for resizing and rescaling
            dataloader_config["transform"] = copy.deepcopy(transform_config)

            # Update dataset directory for the custom dataset
            if "dataset" in dataloader_config.keys() and "ImageFolder" in dataloader_config["dataset"].keys():
                dataloader_config["dataset"]["ImageFolder"]["root"] = dataset.dataset_dir

            dataloader_config["batch_size"] = batch_size

        if "tuning" in config_template.keys():
            config_template["tuning"]["accuracy_criterion"]["relative"] = accuracy_criterion_relative

            if exit_policy_timeout is None:
                config_template["tuning"]["exit_policy"].pop('timeout', None)
            else:
                config_template["tuning"]["exit_policy"]["timeout"] = exit_policy_timeout

            if exit_policy_max_trials is None:
                config_template["tuning"]["exit_policy"].pop('max_trials', None)
            else:
                config_template["tuning"]["exit_policy"]["max_trials"] = exit_policy_max_trials

            if tuning_random_seed is None:
                config_template["tuning"].pop('random_seed', None)
            else:
                config_template["tuning"]["random_seed"] = tuning_random_seed

            if tuning_workspace:
                if "workspace" not in config_template["tuning"].keys():
                    config_template["tuning"]["workspace"] = {}

                config_template["tuning"]["workspace"]["path"] = tuning_workspace
            else:
                # No tuning_workspace is defined, so remove it from the config
                if "workspace" in config_template["tuning"].keys():
                    config_template["tuning"]["workspace"].pop("path", None)

                    if len(config_template["tuning"]["workspace"].keys()) == 0:
                        config_template["tuning"].pop("workspace", None)

        # Create the directory where the file will be written, if it doesn't already exist
        if not os.path.exists(os.path.dirname(config_file_path)):
            os.makedirs(os.path.dirname(config_file_path))

        # Write the config file
        with open(config_file_path, "w") as config_file:
            yaml.dump(config_template, config_file)

    def post_training_quantization(self, saved_model_dir, output_dir, inc_config_path):
        """
        Performs post training quantization using the Intel Neural Compressor on the model from the saved_model_dir
        using the specified config file. The quantized model is written to the output directory.
        
        Args:
            saved_model_dir (str): Source directory for the model to quantize.
            output_dir (str): Writable output directory to save the quantized model
            inc_config_path (str): Path to an INC config file (.yaml)

        Returns:
            None

        Raises:
            NotADirectoryError if the saved_model_dir is not a directory
            FileNotFoundError if a saved_model.pb is not found in the saved_model_dir or if the inc_config_path file
            is not found.
            FileExistsError if the output_dir already has a saved_model.pb file
        """
        # The saved model directory should exist and contain a saved_model.pb file
        if not os.path.isdir(saved_model_dir):
            raise NotADirectoryError("The saved model directory ({}) does not exist.".format(saved_model_dir))
        if not os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb")):
            raise FileNotFoundError("The saved model directory ({}) should have a saved_model.pb file".format(
                saved_model_dir))

        # Verify that the config file exists
        if not os.path.isfile(inc_config_path):
            raise FileNotFoundError("The config file was not found at: {}".format(inc_config_path))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Verify that the output directory doesn't already have a saved_model.pb file
            if os.path.exists(os.path.join(output_dir, "saved_model.pb")):
                raise FileExistsError("A saved model already exists at:", os.path.join(output_dir, "saved_model.pb"))

        from neural_compressor.experimental import Quantization

        quantizer = Quantization(inc_config_path)
        quantizer.model = saved_model_dir
        quantized_model = quantizer.fit()

        # If quantization was successful, save the model
        if quantized_model:
            quantized_model.save(output_dir)

    def benchmark(self, saved_model_dir, inc_config_path, mode='performance'):
        """
        Use INC to benchmark the specified model for performance or accuracy.
        
        Args:
            saved_model_dir (str): Path to the directory where the saved model is located
            inc_config_path (str): Path to an INC config file (.yaml)
            mode (str): performance or accuracy (defaults to performance)

        Returns:
            None

        Raises:
            NotADirectoryError if the saved_model_dir is not a directory
            FileNotFoundError if a saved_model.pb is not found in the saved_model_dir or if the inc_config_path file
            is not found.
            ValueError if an unexpected mode is provided
        """
        # The saved model directory should exist and contain a saved_model.pb file
        if not os.path.isdir(saved_model_dir):
            raise NotADirectoryError("The saved model directory ({}) does not exist.".format(saved_model_dir))
        if not os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb")):
            raise FileNotFoundError("The saved model directory ({}) should have a saved_model.pb file".format(
                saved_model_dir))

        # Validate mode
        if mode not in ['performance', 'accuracy']:
            raise ValueError("Invalid mode: {}. Expected mode to be 'performance' or 'accuracy'.".format(mode))

        # Verify that the config file exists
        if not os.path.isfile(inc_config_path):
            raise FileNotFoundError("The config file was not found at: {}".format(inc_config_path))

        from neural_compressor.experimental import Benchmark

        evaluator = Benchmark(inc_config_path)
        evaluator.model = saved_model_dir
        return evaluator(mode)
