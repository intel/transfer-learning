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

import os

from downloader.models import ModelDownloader
from tlt import TLT_BASE_DIR
from tlt.models.image_classification.pytorch_image_classification_model import PyTorchImageClassificationModel
from tlt.models.image_classification.torchvision_image_classification_model import TorchvisionImageClassificationModel
from tlt.utils.file_utils import read_json_file


class PyTorchHubImageClassificationModel(TorchvisionImageClassificationModel):
    """
    Class to represent a PyTorch Hub pretrained model for image classification
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Class constructor
        """
        pytorch_hub_model_map = read_json_file(os.path.join(
            TLT_BASE_DIR, "models/configs/pytorch_hub_image_classification_models.json"))
        if model_name not in pytorch_hub_model_map.keys():
            raise ValueError("The specified Pytorch Hub image classification model ({}) "
                             "is not supported.".format(model_name))

        PyTorchImageClassificationModel.__init__(self, model_name, **kwargs)

        self._classification_layer = pytorch_hub_model_map[model_name]["classification_layer"]
        self._image_size = pytorch_hub_model_map[model_name]["image_size"]
        self._repo = pytorch_hub_model_map[model_name]["repo"]

        # placeholder for model definition
        self._model = None
        self._num_classes = None
        self._distributed = False

    def _model_downloader(self, model_name):
        downloader = ModelDownloader(model_name, hub='pytorch_hub', model_dir=None)
        model = downloader.download()
        return model
