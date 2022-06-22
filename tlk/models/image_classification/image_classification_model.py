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
import os
import yaml

from tlk import TLK_BASE_DIR
from tlk.models.model import BaseModel
from tlk.utils.types import FrameworkType, UseCaseType


class ImageClassificationModel(BaseModel):
    """
    Class used to represent a pretrained model for image classification
    """

    def __init__(self, image_size, do_fine_tuning: bool, dropout_layer_rate: int,
                 model_name: str, framework: FrameworkType, use_case: UseCaseType):
        self._image_size = image_size
        self._do_fine_tuning = do_fine_tuning
        self._dropout_layer_rate = dropout_layer_rate

        BaseModel.__init__(self, model_name, framework, use_case)

    @property
    def image_size(self):
        return self._image_size

    @property
    @abc.abstractmethod
    def num_classes(self):
        pass

    @property
    def do_fine_tuning(self):
        return self._do_fine_tuning

    @property
    def dropout_layer_rate(self):
        return self._dropout_layer_rate

    def get_inc_config_template_dict(self):
        """
        Returns a dictionary for an config template compatible with the Intel Neural Compressor. It loads the yaml
        file tlk/models/configs/inc/image_classification_template.yaml and then fills in parameters that the model
        knows about (like framework and model name). There are still more parameters that need to be filled in before
        using the config with INC (like the dataset information, image size, etc).
        """
        template_file_path = os.path.join(TLK_BASE_DIR, "models/configs/inc/image_classification_template.yaml")

        if not os.path.exists(template_file_path):
            raise FileNotFoundError("Unable to find the image recognition config template at:", template_file_path)

        with open(template_file_path, 'r') as template_yaml:
            config_template = yaml.safe_load(template_yaml)

        # Update parameters that we know in the template
        config_template["model"]["framework"] = str(self.framework)
        config_template["model"]["name"] = self.model_name

        return config_template
