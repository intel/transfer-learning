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

from enum import Enum, auto


class UseCaseType(Enum):
    IMAGE_CLASSIFICATION = auto()
    TEXT_CLASSIFICATION = auto()
    IMAGE_ANOMALY_DETECTION = auto()
    TEXT_GENERATION = auto()

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(use_case_str):
        use_case_str = use_case_str.lower()

        if use_case_str in ["image_classification", "image classification"]:
            return UseCaseType.IMAGE_CLASSIFICATION
        elif use_case_str in ["text_classification", "text classification"]:
            return UseCaseType.TEXT_CLASSIFICATION
        elif use_case_str in ["anomaly_detection", "anomaly detection",
                              "image_anomaly_detection", "image anomaly detection"]:
            return UseCaseType.IMAGE_ANOMALY_DETECTION
        elif use_case_str in ["text_generation", "text generation"]:
            return UseCaseType.TEXT_GENERATION
        else:
            options = [e.name for e in UseCaseType]
            raise ValueError("Unsupported use case: {} (Select from: {})".format(
                use_case_str, options))


class FrameworkType(Enum):
    TENSORFLOW = auto()
    PYTORCH = auto()

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(framework_str):
        if framework_str.lower() == "tensorflow":
            return FrameworkType.TENSORFLOW
        elif framework_str.lower() == "pytorch":
            return FrameworkType.PYTORCH
        else:
            options = [e.name for e in FrameworkType]
            raise ValueError("Unsupported framework: {} (Select from: {})".format(
                framework_str, options))
