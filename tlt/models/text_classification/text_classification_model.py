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

from tlt.models.model import BaseModel
from tlt.utils.types import FrameworkType, UseCaseType


class TextClassificationModel(BaseModel):
    """
    Class used to represent a pretrained model for text classification
    """

    def __init__(self, model_name: str, framework: FrameworkType, use_case: UseCaseType, dropout_layer_rate: float):
        self._dropout_layer_rate = dropout_layer_rate
        BaseModel.__init__(self, model_name, framework, use_case)

        # Default learning rate for text models
        self._learning_rate = 3e-5

    @property
    @abc.abstractmethod
    def num_classes(self):
        pass

    @property
    def dropout_layer_rate(self):
        return self._dropout_layer_rate

    def write_inc_config_file(self, config_file_path, dataset, batch_size, overwrite=False, **kwargs):
        raise NotImplementedError("Writing INC config files has not be implemented yet for text classification")

    def quantize(self, saved_model_dir, output_dir, inc_config_path):
        raise NotImplementedError("Post training quantization has not been implemented yet for text classification")

    def optimize_graph(self, saved_model_dir, output_dir):
        raise NotImplementedError("Optimize graph has not been implemented yet for text classification")

    def benchmark(self, saved_model_dir, inc_config_path, mode='performance'):
        raise NotImplementedError("Benchmarking has not been implemented yet for text classification")
