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

from tlk.utils.types import FrameworkType
from tlk.models.image_classification.tfhub_image_classification_model import TFHubImageClassificationModel


def get_model(model_name: str, framework: FrameworkType):
    if not isinstance(framework, FrameworkType):
        framework = FrameworkType.from_str(framework)

    if framework == FrameworkType.PYTORCH:
        raise NotImplementedError("PyTorch support has not been implemented")

    # TODO: Support other model types and support passing extra configs
    return TFHubImageClassificationModel(model_name)
