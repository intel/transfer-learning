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
import dill
import numpy
import random
import torch

from tlt.models.model import BaseModel
from tlt.utils.types import FrameworkType, UseCaseType
from tlt.utils.file_utils import verify_directory


class PyTorchModel(BaseModel):
    """
    Base class used to represent a PyTorch model
    """

    def __init__(self, model_name: str, framework: FrameworkType, use_case: UseCaseType):
        """
        Class constructor
        """
        super().__init__(model_name, framework, use_case)
        self._lr_scheduler = None
        self._history = {}

        # Setup warnings module to set warnings to go to stdout
        import warnings
        import sys

        def customwarn(message, category, filename, lineno, file=None, line=None):
            sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))
        warnings.showwarning = customwarn

    def _set_seed(self, seed):
        if seed is not None:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)

    def _check_train_inputs(self, output_dir, dataset, dataset_type, epochs, initial_checkpoints):
        verify_directory(output_dir)

        if not isinstance(dataset, dataset_type):
            raise TypeError("The dataset must be a {} but found a {}".format(dataset_type, type(dataset)))

        if not isinstance(epochs, int):
            raise TypeError("Invalid type for the epochs arg. Expected an int but found a {}".format(type(epochs)))

        if initial_checkpoints and not isinstance(initial_checkpoints, str):
            raise TypeError("The initial_checkpoints parameter must be a string but found a {}".format(
                type(initial_checkpoints)))

    def _update_history(self, key, value):
        if key not in self._history:
            self._history[key] = []
        self._history[key].extend([value])

    def load_from_directory(self, model_dir: str):
        """
        Load a saved model from the model_dir path
        """
        # Verify that the model directory exists
        verify_directory(model_dir, require_directory_exists=True)
        model_copy = torch.load(os.path.join(model_dir, 'model.pt'))
        self._model = dill.loads(model_copy)
        self._optimizer = self._optimizer_class(self._model.parameters(), lr=self._learning_rate)

    def write_inc_config_file(self, config_file_path, dataset, batch_size, overwrite=False, **kwargs):
        """
        Writes an INC compatible config file to the specified path usings args from the specified dataset and
        parameters. This is currently only supported for TF custom image classification datasets.

           Args:
               config_file_path (str): Destination path on where to write the .yaml config file.
               dataset (BaseDataset): A tlt dataset object
               batch_size (int): Batch size to use for quantization and evaluation
               overwrite (bool): Specify whether or not to overwrite the config_file_path, if it already exists
                                 (default: False)

           Returns:
               None

           Raises:
               NotImplementedError because this hasn't been implemented yet for PyTorch
           """
        raise NotImplementedError("Writing an INC config file is not supported PyTorch models yet")

    def optimize_graph(self, saved_model_dir, output_dir):
        """
        Performs FP32 graph optimization using the Intel Neural Compressor on the model in the saved_model_dir
        and writes the inference-optimized model to the output_dir. Graph optimization includes converting
        variables to constants, removing training-only operations like checkpoint saving, stripping out parts
        of the graph that are never reached, removing debug operations like CheckNumerics, folding batch
        normalization ops into the pre-calculated weights, and fusing common operations into unified versions.

        Args:
            saved_model_dir (str): Source directory for the model to optimize.
            output_dir (str): Writable output directory to save the optimized model

        Returns:
            None

        Raises:
            NotImplementedError because this hasn't been implemented yet for PyTorch
        """
        raise NotImplementedError("Only TensorFlow graph optimization is currently supported by the \
                                                                      Intel Neural Compressor (INC)")

    def quantize(self, saved_model_dir, output_dir, inc_config_path):
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
            NotImplementedError because this hasn't been implemented yet for PyTorch
        """
        raise NotImplementedError("Quantization is not supported for PyTorch models in tlt")

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
            NotImplementedError because this hasn't been implemented yet for PyTorch
        """
        raise NotImplementedError("INC benchmarking is not supported for PyTorch models in tlt")
