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

import os
import inspect
import torch

from neural_compressor import quantization
from neural_compressor.config import BenchmarkConfig

from tlt.models.model import BaseModel
from tlt.utils.types import FrameworkType, UseCaseType
from tlt.utils.file_utils import verify_directory
from tlt.utils.inc_utils import get_inc_config


class HFModel(BaseModel):
    """
    Base class to represent a Hugging Face model
    """

    def __init__(self, model_name: str, framework: FrameworkType, use_case: UseCaseType):
        super().__init__(model_name, framework, use_case)
        self._history = {}

    def _update_history(self, key, value):
        if key not in self._history:
            self._history[key] = []
        self._history[key].extend([value])

    def _check_optimizer_loss(self, optimizer, loss):
        if optimizer is not None and (not inspect.isclass(optimizer) or
                                      torch.optim.Optimizer not in inspect.getmro(optimizer)):
            raise TypeError("The optimizer input must be a class (not an instance) of type torch.optim.Optimizer or "
                            "None but found a {}. Example: torch.optim.AdamW".format(type(optimizer)))
        if loss is not None and (not inspect.isclass(loss) or
                                 torch.nn.modules.loss._Loss not in inspect.getmro(loss)):
            raise TypeError("The optimizer input must be a class (not an instance) of type "
                            "torch.nn.modules.loss._Loss or None but found a {}. "
                            "Example: torch.nn.CrossEntropyLoss".format(type(loss)))

    def _check_train_inputs(self, output_dir, dataset, dataset_type, extra_layers, epochs, distributed, hostfile):
        verify_directory(output_dir)

        if distributed:
            if hostfile is not None and not os.path.exists(os.path.join(os.getcwd(), hostfile)):
                raise FileNotFoundError("Could not find hostfile. Consider creating one")

        if not isinstance(dataset, dataset_type):
            raise TypeError("The dataset must be a {} but found a {}".format(dataset_type, type(dataset)))

        if not dataset.info['preprocessing_info']:
            raise ValueError("Dataset hasn't been preprocessed yet.")

        if extra_layers:
            if not isinstance(extra_layers, list) or not all(isinstance(n, int) for n in extra_layers):
                raise ValueError("extra_layers argument must be a list of integers but found a {}".format(extra_layers))

        if not isinstance(epochs, int):
            raise TypeError("Invalid type for the epochs arg. Expected an int but found a {}".format(type(epochs)))

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
            NotImplementedError: because this hasn't been implemented yet for PyTorch
        """
        raise NotImplementedError("Only TensorFlow graph optimization is currently supported by the "
                                  "Intel Neural Compressor (INC)")

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
            FileExistsError: if the output_dir already has a model.pt file
            ValueError: if the dataset is not compatible for quantizing the model
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Verify that the output directory doesn't already have a model.pt or best_model.pt file
            if os.path.exists(os.path.join(output_dir, "model.pt")) or \
                    os.path.exists(os.path.join(output_dir, "best_model.pt")):
                if not overwrite_model:
                    raise FileExistsError("A saved model already exists in: {}".format(output_dir))

        # Verify dataset is of the right type
        if not isinstance(dataset, self._inc_compatible_dataset):
            raise ValueError('Quantization is compatible with datasets of type {}, and type '
                             '{} was found'.format(self._inc_compatible_dataset, type(dataset)))

        config = config if config is not None else get_inc_config(approach=self._quantization_approach)

        calib_dataloader, _ = dataset.get_inc_dataloaders()
        config.backend = 'default'
        quantized_model = quantization.fit(model=self._model, conf=config, calib_dataloader=calib_dataloader)

        # If quantization was successful, save the model
        if quantized_model:
            quantized_model.save(output_dir)
            if os.path.isfile(os.path.join(output_dir, 'best_model.pt')):
                # Change the model filename from best_model.pt to model.pt to match our convention
                os.rename(os.path.join(output_dir, 'best_model.pt'), os.path.join(output_dir, 'model.pt'))
        else:
            raise RuntimeError("There was an error with quantization")

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
            FileNotFoundError: if a model.pt is not found in the saved_model_dir or if the inc_config_path file
            is not found
        """
        # Verify dataset is of the right type
        if not isinstance(dataset, self._inc_compatible_dataset):
            raise NotImplementedError('Quantization has only been implemented for TLT datasets, and type '
                                      '{} was found'.format(type(dataset)))

        # If provided, the saved model directory should exist and contain a model.pt file
        if saved_model_dir is not None:
            if not os.path.isdir(saved_model_dir):
                raise NotADirectoryError("The saved model directory ({}) does not exist.".format(saved_model_dir))
            if not os.path.isfile(os.path.join(saved_model_dir, "model.pt")):
                raise FileNotFoundError("The saved model directory ({}) should have a model.pt file".format(
                    saved_model_dir))
            model = os.path.join(saved_model_dir, 'model.pt')
        else:
            model = self._model

        config = BenchmarkConfig(backend="ipex", warmup=warmup, iteration=iteration,
                                 cores_per_instance=cores_per_instance, num_of_instance=num_of_instance,
                                 inter_num_of_threads=inter_num_of_threads, intra_num_of_threads=intra_num_of_threads)

        _, eval_dataloader = dataset.get_inc_dataloaders()

        from neural_compressor.benchmark import fit

        try:
            return fit(model=model, config=config, b_dataloader=eval_dataloader)
        except AssertionError:
            # Use INC's special load utility to reload an int8 ipex model
            from neural_compressor.utils.pytorch import load
            quantized_model = load(model, self._model, dataloader=eval_dataloader)
            return fit(model=quantized_model, config=config, b_dataloader=eval_dataloader)
