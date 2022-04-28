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


from tlk.datasets import dataset_factory
from tlk.datasets.image_classification.tf_image_classification_dataset import TFImageClassificationDataset

# TODO: Create a fixture to initialize dataset and delete it once all tests have run
def test_tf_flowers():
    """
    Checks that a tf_flowers dataset can be created/loaded
    """
    flowers = dataset_factory.get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers',
                                          'tf_datasets')
    assert type(flowers) == TFImageClassificationDataset
    assert len(flowers.class_names) == 5
    assert len(flowers.dataset) == 2752

def test_tf_flowers_10pct():
    """
    Checks that a 10% tf_flowers subset can be loaded
    """
    flowers = dataset_factory.get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers',
                                          'tf_datasets', split=["train[:10%]"])
    assert type(flowers) == TFImageClassificationDataset
    assert len(flowers.dataset) < 2752
