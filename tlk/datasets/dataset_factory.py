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

from tlk.utils.types import FrameworkType, UseCaseType
from tlk.datasets.image_classification.tf_image_classification_dataset import TFImageClassificationDataset

dataset_map = {
    FrameworkType.TENSORFLOW: {
        UseCaseType.IMAGE_CLASSIFICATION: {
            "tf_datasets": TFImageClassificationDataset
        }
    }
}


def get_dataset(dataset_dir: str, use_case: UseCaseType, framework: FrameworkType,
                dataset_name=None, dataset_catalog=None):
    if not isinstance(framework, FrameworkType):
        framework = FrameworkType.from_str(framework)

    if not isinstance(use_case, UseCaseType):
        use_case = UseCaseType.from_str(use_case)

    if framework in dataset_map.keys():
        if use_case in dataset_map[framework].keys():
            if dataset_catalog and dataset_catalog in dataset_map[framework][use_case]:
                return dataset_map[framework][use_case][dataset_catalog](dataset_dir, dataset_name)

    # For the error message, if there's no dataset catalog specified, then it's a custom dataset
    if not dataset_catalog:
        dataset_catalog = "custom datasets"

    # If no match was found in the map, then it's not implemented yet
    raise NotImplementedError("Datasets support for {} {} {} has not been implemented yet".format(
        str(framework), str(use_case), dataset_catalog))
