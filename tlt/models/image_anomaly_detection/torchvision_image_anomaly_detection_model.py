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
# SPDX-License-Identifier: EPL-2.0
#

from pydoc import locate

import torch
from torchvision.models.feature_extraction import create_feature_extractor

from tlt.models.image_classification.torchvision_image_classification_model import TorchvisionImageClassificationModel


class TorchvisionImageAnomalyDetectionModel(TorchvisionImageClassificationModel):
    """
    Class to represent a Torchvision pretrained model for anomaly detection
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Class constructor
        """
        TorchvisionImageClassificationModel.__init__(self, model_name, **kwargs)

    def extract_features(self, data, layer_name, pooling):
        """
        Extracts features of the layers specified using a layer name
        Args:
            data (MVTech Dataset): Dataset to use when extracting features
            layer_name (string): The layer name that will be frozen in the model
            pooling (list[string, int]): Pooling to be applied on the extracted layer

        Returns:
            outputs : Extracted features after applying pooling

        Raises:
            ValueError if the parameters are not within the expected values
        """

        pretrained_model_class = locate('torchvision.models.{}'.format(self.model_name))
        model = pretrained_model_class(pretrained=True)

        model.eval()
        return_nodes = {layer: layer for layer in [layer_name]}
        partial_model = create_feature_extractor(model, return_nodes=return_nodes)
        features = partial_model(data)[layer_name]
        pooling_list = ['avg', 'max']
        if pooling[0] in pooling_list and pooling[0] == 'avg':
            pool_out = torch.nn.functional.avg_pool2d(features, pooling[1])
        elif pooling[0] in pooling_list and pooling[0] == 'max':
            pool_out = torch.nn.functional.max_pool2d(features, pooling[1])
        else:
            raise ValueError("The specified pooling is not supported")
        outputs = pool_out.contiguous().view(pool_out.size(0), -1)

        return outputs
