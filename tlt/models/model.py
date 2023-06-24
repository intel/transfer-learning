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

import abc

from tlt.utils.types import FrameworkType, UseCaseType
from tlt.datasets.dataset import BaseDataset


class BaseModel(abc.ABC):
    """
    Abstract base class for a pretrained model that can be used for transfer learning
    """

    def __init__(self, model_name: str, framework: FrameworkType, use_case: UseCaseType):
        """
        Class constructor
        """
        self._model_name = model_name
        self._framework = framework
        self._use_case = use_case
        self._learning_rate = 0.001
        self._preprocessor = None

    @property
    def model_name(self):
        """
        Name of the model
        """
        return self._model_name

    @property
    def framework(self):
        """
        Framework with which the model is compatible
        """
        return self._framework

    @property
    def use_case(self):
        """
        Use case (or category) to which the model belongs
        """
        return self._use_case

    @property
    def learning_rate(self):
        """
        Learning rate for the model
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    def preprocessor(self):
        """
        Preprocessor for the model
        """
        return self._preprocessor

    @abc.abstractmethod
    def load_from_directory(self, model_dir: str):
        """
        Load a model from a directory
        """
        pass

    @abc.abstractmethod
    def train(self, dataset: BaseDataset, output_dir, epochs=1, initial_checkpoints=None, do_eval=True):
        """
        Train the model using the specified dataset
        """
        pass

    @abc.abstractmethod
    def evaluate(self, dataset: BaseDataset):
        """
        Evaluate the model using the specified dataset.

        Returns the loss and metrics for the model in test mode.
        """
        pass

    @abc.abstractmethod
    def predict(self, input_samples):
        """
        Generates predictions for the input samples.

        The input samples can be a BaseDataset type of object or a numpy array.
        Returns a numpy array of predictions.
        """
        pass

    @abc.abstractmethod
    def export(self, output_dir: str):
        """
        Export the serialized model to an output directory
        """
        pass

    @abc.abstractmethod
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
            FileExistsError: if the output_dir already has a model file
            ValueError: if the dataset is not compatible for quantizing the model
        """
        pass

    @abc.abstractmethod
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
            NotImplementedError: if the model does not support INC yet
            FileExistsError: if the output_dir already has a saved_model.pb file
        """
        pass

    @abc.abstractmethod
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
            NotADirectoryError: if the saved_model_dir is not None or a valid directory
            FileNotFoundError: if a model is not found in the saved_model_dir
        """
        raise NotImplementedError("INC benchmarking is not supported for this model")
