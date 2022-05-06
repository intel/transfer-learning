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

from tlk.utils.types import FrameworkType, UseCaseType
from tlk.datasets.dataset import BaseDataset


class BaseModel(abc.ABC):
    """
    Abstract base class for a pretrained model that can be used for transfer learning
    """

    def __init__(self, model_name: str, framework: FrameworkType, use_case: UseCaseType):
        self._model_name = model_name
        self._framework = framework
        self._use_case = use_case

    @property
    def model_name(self):
        return self._model_name

    @property
    def framework(self):
        return self._framework

    @property
    def use_case(self):
        return self._use_case

    @abc.abstractmethod
    def load_from_directory(self, model_dir: str):
        """
        Loads a model from a directory
        """
        pass

    @abc.abstractmethod
    def train(self, dataset: BaseDataset, output_dir, epochs=1):
        """
        Train the model using the specified dataset
        """
        pass

    @abc.abstractmethod
    def evaluate(self, dataset: BaseDataset):
        """
        Evaluate the model using the specified dataset. Returns the loss and metrics for the model in test mode.
        """
        pass

    @abc.abstractmethod
    def predict(self, input_samples):
        """
        Generates predictions for the input samples. The input samples can be a BaseDataset type of object or a numpy
        array. Returns a numpy array of predictions.
        """
        pass

    @abc.abstractmethod
    def export(self, output_dir: str):
        pass
