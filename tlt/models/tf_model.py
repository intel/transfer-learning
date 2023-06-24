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
# SPDX-License-Identifier: Apache-2.0
#

import inspect
import os
import dill  # nosec: B403
import re
import shutil
import random
import tempfile
import numpy as np
import tensorflow as tf

from neural_compressor.experimental import Graph_Optimization
from neural_compressor import quantization
from neural_compressor.config import BenchmarkConfig

from tlt.models.model import BaseModel
from tlt.models.text_classification.text_classification_model import TextClassificationModel
from tlt.utils.file_utils import verify_directory, validate_model_name
from tlt.utils.platform_util import PlatformUtil
from tlt.utils.types import FrameworkType, UseCaseType
from tlt.utils.inc_utils import get_inc_config


class TFModel(BaseModel):
    """
    Base class to represent a TF pretrained model
    """

    def __init__(self, model_name: str, framework: FrameworkType, use_case: UseCaseType):
        self._model = None
        super().__init__(model_name, framework, use_case)
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
        self._history = {}

    def _set_seed(self, seed):
        if seed is not None:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

    def _check_train_inputs(self, output_dir, dataset, dataset_type, epochs, initial_checkpoints):
        verify_directory(output_dir)

        if not isinstance(dataset, dataset_type):
            raise TypeError("The dataset must be a {} but found a {}".format(dataset_type, type(dataset)))

        if not isinstance(epochs, int):
            raise TypeError("Invalid type for the epochs arg. Expected an int but found a {}".format(type(epochs)))

        if initial_checkpoints and not isinstance(initial_checkpoints, str):
            raise TypeError("The initial_checkpoints parameter must be a string but found a {}".format(
                type(initial_checkpoints)))

    def _check_optimizer_loss(self, optimizer, loss):
        if optimizer is not None and (not inspect.isclass(optimizer) or
                                      tf.keras.optimizers.Optimizer not in inspect.getmro(optimizer)):
            raise TypeError("The optimizer input must be a class (not an instance) of type "
                            "tf.keras.optimizers.Optimizer or None but found a {}. "
                            "Example: tf.keras.optimizers.SGD".format(optimizer))
        if loss is not None and (not inspect.isclass(loss) or
                                 tf.keras.losses.Loss not in inspect.getmro(loss)):
            raise TypeError("The loss input must be class (not an instance) of type "
                            "tf.keras.losses.Loss or None but found a {}. "
                            "Example: tf.keras.losses.BinaryCrossentropy".format(loss))

    def load_from_directory(self, model_dir: str):
        """
            Loads a saved model from the specified directory

            Args:
                model_dir (str): Directory with a saved_model.pb or h5py file to load

            Returns:
                None

            Raises:
                TypeError: if model_dir is not a string
                NotADirectoryError: if model_dir is not a directory
                IOError: for an invalid model file
        """
        # Verify that the model directory exists
        verify_directory(model_dir, require_directory_exists=True)

        self._model = tf.keras.models.load_model(model_dir)
        self._model.summary(print_fn=print)

    def set_auto_mixed_precision(self, enable_auto_mixed_precision):
        """
        Enable auto mixed precision for training. Mixed precision uses both 16-bit and 32-bit floating point types to
        make training run faster and use less memory. If enable_auto_mixed_precision is set to None, auto mixed
        precision will be enabled when running with Intel fourth generation Xeon processors, and disabled for other
        platforms.
        """
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
                enable_auto_mixed_precision = PlatformUtil().cpu_type == 'SPR'
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

    def export(self, output_dir):
        """
           Exports a trained model as a saved_model.pb file. The file will be written to the output directory in a
           directory with the model's name, and a unique numbered directory (compatible with TF serving). The directory
           number will increment each time the model is exported.

           Args:
               output_dir (str): A writeable output directory.

           Returns:
               The path to the numbered saved model directory

           Raises:
               TypeError: if the output_dir is not a string
               FileExistsError: the specified output directory already exists as a file
               ValueError: if the mode has not been loaded or trained yet
        """
        if self._model:
            # Save the model in a format that can be served
            verify_directory(output_dir)
            val_model_name = validate_model_name(self.model_name)
            saved_model_dir = os.path.join(output_dir, val_model_name)
            if os.path.exists(saved_model_dir) and len(os.listdir(saved_model_dir)):
                saved_model_dir = os.path.join(saved_model_dir, "{}".format(len(os.listdir(saved_model_dir)) + 1))
            else:
                saved_model_dir = os.path.join(saved_model_dir, "1")

            self._model.save(saved_model_dir)
            print("Saved model directory:", saved_model_dir)

            return saved_model_dir
        else:
            raise ValueError("Unable to export the model, because it hasn't been loaded or trained yet")

    def export_for_distributed(self, export_dir=None, train_data=None, val_data=None):
        """
        Exports the model, optimizer, loss, train data and validation data to the export_dir for distributed
        script to access. Note that the export_dir must be accessible to all the nodes. For example: NFS shared
        systems. Note that the export_dir is created using mkdtemp which reults in a unique dir name. For
        example: "<export_dir_Am83Iw". If the export_dir is None, the default name is "saved_objects"

        Args:
            export_dir (str): Directory name to export the model, optimizer, loss, train data and validation
                data. export_dir must be accessible to all the nodes. For example: NFS shared systems. export_dir
                is created using mkdtemp which reults in a unique dir name. Forexample: "<export_dir_Am83Iw".
                If the export_dir is None, the default name is "saved_objects"
            train_data (TFDataset): Train dataset
            val_data (TFDataset): Validation dataset
        """

        temp_dir_prefix = os.path.join(os.environ['HOME'], "saved_objects_") if export_dir is None else export_dir + "_"
        self._temp_dir = tempfile.mkdtemp(prefix=temp_dir_prefix)

        # Save the model
        print('Saving the model...', end='', flush=True)
        tf.keras.models.save_model(
            model=self._model,
            filepath=self._temp_dir,
            overwrite=True,
            include_optimizer=False
        )
        print('Done')

        # Save the optimizer object
        print('Saving the optimizer...', end='', flush=True)
        tf.train.Checkpoint(optimizer=self._optimizer).save(
            os.path.join(self._temp_dir, 'saved_optimizer'))
        print('Done')

        # Save the loss class name and its args
        print('Saving the loss...', end='', flush=True)
        with open(os.path.join(self._temp_dir, 'saved_loss'), 'wb') as f:
            dill.dump((self._loss_class, self._loss_args), f)
            print('Done')

        # Save the dataset(s)
        print('Saving the train data...', end='', flush=True)
        train_data.save(os.path.join(self._temp_dir, 'train_data'))
        print('Done')
        if val_data:
            print('Saving the validation data...', end='', flush=True)
            val_data.save(os.path.join(self._temp_dir, 'val_data'))
            print('Done')
        return self._temp_dir

    def cleanup_saved_objects_for_distributed(self):
        try:
            print('Cleaning saved objects...')
            shutil.rmtree(self._temp_dir)
        except OSError as ose:
            print('Error while cleaning the saved obects: {}'.format(ose))

    def _parse_hostfile(self, hostfile):
        """
            Parses the hostfile and returns the required command. Contents of hostfile must contain IP addresses
            (or) hostnames in any of the following forms. Note that all lines must be of the same form:
                "127.0.0.1"
                "127.0.0.1 slots=2"
                "127.0.0.1:2"
                "hostname-example.com"
                "hostname-example.com slots=2"
                "hostname-example.com:2"

            Args:
                hostfile (str): File path of the hostfile

            Returns:
                hostfile_info dict containing ip_addresses and slots

        """
        import socket
        valid_regexes = {
            'valid_ip': r"^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])$",  # noqa: E501
            'valid_ip_with_slot': r"^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9]) slots=[0-9]{1,2}$",  # noqa: E501
            'valid_ip_with_colon': r"^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9]):\d{1,2}$",  # noqa: E501
            'valid_hostname': r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$",  # noqa: E501
            'valid_hostname_with_slot': r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]) slots=[0-9]{1,2}$",  # noqa: E501
            'valid_hostname_with_colon': r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]):\d{1,2}$"  # noqa: E501
        }
        lines = []
        with open(hostfile, 'r') as f:
            lines = [line.rstrip() for line in f]

        matches = []
        for line in lines:
            found = False
            for k, v in valid_regexes.items():
                if re.match(v, line):
                    found = True
                    matches.append(k)
                    break
            if not found:
                raise ValueError("Invalid line in the hostfile: {}".format(line))

        hostfile_info = {
            'ip_addresses': [],
            'slots': []
        }
        for line, match in zip(lines, matches):
            if match == 'valid_ip':
                hostfile_info['ip_addresses'].append(line)
                hostfile_info['slots'].append(0)
            elif match == 'valid_ip_with_slot':
                hostfile_info['ip_addresses'].append(line.split(' slots=')[0])
                hostfile_info['slots'].append(line.split(' slots=')[1])
            elif match == 'valid_ip_with_colon':
                hostfile_info['ip_addresses'].append(line.split(':')[0])
                hostfile_info['slots'].append(line.split(':')[1])
            elif match == 'valid_hostname':
                hostfile_info['ip_addresses'].append(socket.gethostbyname(line))
                hostfile_info['slots'].append(0)
            elif match == 'valid_hostname_with_slot':
                hostfile_info['ip_addresses'].append(socket.gethostbyname(line.split(' slots=')[0]))
                hostfile_info['slots'].append(line.split(' slots=')[1])
            elif match == 'valid_hostname_with_colon':
                hostfile_info['ip_addresses'].append(socket.gethostbyname(line.split(':')[0]))
                hostfile_info['slots'].append(line.split(':')[1])

        return hostfile_info

    def optimize_graph(self, output_dir, overwrite_model=False):
        """
        Performs FP32 graph optimization using the Intel Neural Compressor on the model
        and writes the inference-optimized model to the output_dir. Graph optimization includes converting
        variables to constants, removing training-only operations like checkpoint saving, stripping out parts
        of the graph that are never reached, removing debug operations like CheckNumerics, folding batch
        normalization ops into the pre-calculated weights, and fusing common operations into unified versions.

        Args:
            output_dir (str): Writable output directory to save the optimized model
            overwrite_model (bool): Specify whether or not to overwrite the output_dir, if it already exists
                                    (default: False)

        Returns:
            None

        Raises:
            FileExistsError: if the output_dir already has a saved_model.pb file
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Verify that the output directory doesn't already have a saved_model.pb file
            if os.path.exists(os.path.join(output_dir, "saved_model.pb")) and not overwrite_model:
                raise FileExistsError("A saved model already exists at:", os.path.join(output_dir, "saved_model.pb"))

        graph_optimizer = Graph_Optimization()
        graph_optimizer.model = self._model
        optimized_graph = graph_optimizer()

        # If optimization was successful, save the model
        if optimized_graph:
            optimized_graph.save(output_dir)

    def quantize(self, output_dir, dataset, config=None, overwrite_model=False):
        """
        Performs post training quantization using the Intel Neural Compressor on the model using the dataset.
        The dataset's training subset will be used as the calibration data and its validation or test subset will
        be used for evaluation. The quantized model is written to the output directory.

        Args:
            output_dir (str): Writable output directory to save the quantized model
            dataset (ImageClassificationDataset): dataset to quantize with
            config (PostTrainingQuantConfig): Optional, for customizing the quantization parameters
            overwrite_model (bool): Specify whether or not to overwrite the output_dir, if it already exists
                                    (default: False)

        Returns:
            None

        Raises:
            FileExistsError: if the output_dir already has a saved_model.pb file
            ValueError: if the dataset is not compatible for quantizing the model
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Verify that the output directory doesn't already have a saved_model.pb file
            if os.path.exists(os.path.join(output_dir, "saved_model.pb")) and not overwrite_model:
                raise FileExistsError("A saved model already exists at:", os.path.join(output_dir, "saved_model.pb"))

        # Verify dataset is of the right type
        if not isinstance(dataset, self._inc_compatible_dataset):
            raise ValueError('Quantization is compatible with datasets of type {}, and type '
                             '{} was found'.format(self._inc_compatible_dataset, type(dataset)))

        config = config if config is not None else get_inc_config(approach=self._quantization_approach)
        kwargs = {}
        if isinstance(self, TextClassificationModel):
            kwargs['hub_name'] = self._hub_name
            kwargs['max_seq_length'] = self._max_seq_length
        calib_dataloader, eval_dataloader = dataset.get_inc_dataloaders(**kwargs)
        quantized_model = quantization.fit(model=self._model, conf=config, calib_dataloader=calib_dataloader,
                                           eval_dataloader=eval_dataloader)

        # If quantization was successful, save the model
        if quantized_model:
            quantized_model.save(output_dir)

    def benchmark(self, dataset, saved_model_dir=None, warmup=10, iteration=100, cores_per_instance=None,
                  num_of_instance=None, inter_num_of_threads=None, intra_num_of_threads=None):
        """
        Use Intel Neural Compressor to benchmark the model with the dataset argument. The dataset's validation or test
        subset will be used for benchmarking, if present. Otherwise, the full training dataset is used. The model to be
        benchmarked can also be explicitly set to a saved_model_dir containing for example a quantized saved model.

        Args:
            dataset (ImageClassificationDataset): Dataset to use for benchmarking
            saved_model_dir (str): Optional, path to the directory where the saved model is located
            warmup (int): The number of iterations to perform before running performance tests, default is 10
            iteration (int): The number of iterations to run performance tests, default is 100
            cores_per_instance (int or None): The number of CPU cores to use per instance, default is None
            num_of_instance (int or None): The number of instances to use for performance testing, default is None
            inter_num_of_threads (int or None): The number of threads to use for inter-thread operations, default is
                                                None
            intra_num_of_threads (int or None): The number of threads to use for intra-thread operations, default is
                                                None

        Returns:
            Benchmarking results from Intel Neural Compressor

        Raises:
            NotADirectoryError: if the saved_model_dir is not a directory
            FileNotFoundError: if a saved_model.pb is not found in the saved_model_dir or if the inc_config_path file
            is not found
        """
        # If provided, the saved model directory should exist and contain a saved_model.pb file
        if saved_model_dir is not None:
            if not os.path.isdir(saved_model_dir):
                raise NotADirectoryError("The saved model directory ({}) does not exist.".format(saved_model_dir))
            if not os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb")):
                raise FileNotFoundError("The saved model directory ({}) should have a saved_model.pb file".format(
                    saved_model_dir))
            model = saved_model_dir
        else:
            model = self._model

        kwargs = {}
        if isinstance(self, TextClassificationModel):
            kwargs['hub_name'] = self._hub_name
            kwargs['max_seq_length'] = self._max_seq_length
        _, eval_dataloader = dataset.get_inc_dataloaders(**kwargs)
        config = BenchmarkConfig(warmup=warmup, iteration=iteration, cores_per_instance=cores_per_instance,
                                 num_of_instance=num_of_instance, inter_num_of_threads=inter_num_of_threads,
                                 intra_num_of_threads=intra_num_of_threads)

        from neural_compressor.benchmark import fit

        return fit(model, config=config, b_dataloader=eval_dataloader)
