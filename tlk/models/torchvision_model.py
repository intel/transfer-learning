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

import abc

from tlk.models.model import BaseModel
from tlk.utils.types import FrameworkType, UseCaseType


class TorchvisionModel(BaseModel):
    """
    Class used to represent a Torchvision pretrained model
    """

    def __init__(self, model_name: str, framework: FrameworkType, use_case: UseCaseType):
        super().__init__(model_name, framework, use_case)

    def write_inc_config_file(self, config_file_path, dataset, batch_size, overwrite=False, **kwargs):
        """
        Writes an INC compatible config file to the specified path usings args from the specified dataset and
        parameters. This is currently only supported for TF custom image classification datasets.

           Args:
               config_file_path (str): Destination path on where to write the .yaml config file.
               dataset (BaseDataset): A tlk dataset object
               batch_size (int): Batch size to use for quantization and evaluation
               overwrite (bool): Specify whether or not to overwrite the config_file_path, if it already exists
                                 (default: False)

           Returns:
               None

           Raises:
               NotImplementedError because this hasn't been implemented yet for torchvision
           """
        raise NotImplementedError("Writing an INC config file is not supported torchvision models yet")

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
            NotImplementedError because this hasn't been implemented yet for torchvision
        """
        raise NotImplementedError("Quantization is not supported for torchvision models in tlk")

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
            NotImplementedError because this hasn't been implemented yet for torchvision
        """
        raise NotImplementedError("INC benchmarking is not supported for torchvision models in tlk")
