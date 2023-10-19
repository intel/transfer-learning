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

import abc

from tlt.models.model import BaseModel
from tlt.utils.types import FrameworkType, UseCaseType


class TextGenerationModel(BaseModel):
    """
    Class to represent a pretrained model for text generation
    """

    def __init__(self, model_name: str, framework: FrameworkType, use_case: UseCaseType):
        BaseModel.__init__(self, model_name, framework, use_case)

    @abc.abstractmethod
    def generate(self, input_samples):
        """
        Generates text completions for the input samples.

        The input samples can be a string or list of strings.
        Returns a list of strings.
        """
        pass
