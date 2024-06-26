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
from tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model import PyTorchImageAnomalyDetectionModel
from tlt.utils.file_utils import read_json_file


class TorchvisionImageAnomalyDetectionModel(PyTorchImageAnomalyDetectionModel):
    """
    Class to represent a Torchvision pretrained model for anomaly detection
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Class constructor
        """
        PyTorchImageAnomalyDetectionModel.__init__(self, model_name, **kwargs)

        torchvision_model_map = read_json_file(os.path.join(
            TLT_BASE_DIR, "models/configs/torchvision_image_anomaly_detection_models.json"))
        if model_name not in torchvision_model_map.keys():
            raise ValueError("The specified Torchvision image anomaly detection model ({}) "
                             "is not supported.".format(model_name))

        self._image_size = torchvision_model_map[model_name]["image_size"]
        self._original_dataset = torchvision_model_map[model_name]["original_dataset"]
        self._hub = "torchvision"
        self._classification_layer = torchvision_model_map[model_name]["classification_layer"]

        self._model = self._model_downloader(self._model_name)

    def _model_downloader(self, model_name):
        downloader = ModelDownloader(model_name, hub=self._hub, model_dir=None)
        model = downloader.download()
        return model
