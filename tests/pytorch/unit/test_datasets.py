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

import pytest
from numpy.testing import assert_array_equal
from tlk.datasets.dataset_factory import get_dataset

try:
    # Do torch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlk.datasets.image_classification.torchvision_image_classification_dataset import TorchvisionImageClassificationDataset
except ModuleNotFoundError as e:
    print("WARNING: Unable to import TorchvisionImageClassificationDataset. Torch may not be installed")

# TODO: Create a fixture to initialize dataset and delete it once all tests have run
@pytest.mark.pytorch
def test_torchvision():
    """
    Checks that torchvision dataset can be created/loaded
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10', 'torchvision')
    assert type(data) == TorchvisionImageClassificationDataset
    assert len(data.class_names) == 10
    assert data.info['dataset_info'] == {'name': 'CIFAR10','size': 50000}

@pytest.mark.pytorch
def test_torchvision_subset():
    """
    Checks that a torchvision test subset can be loaded
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10', 'torchvision', split=["test"])
    assert type(data) == TorchvisionImageClassificationDataset
    assert len(data.dataset) < 50000

@pytest.mark.pytorch
def test_preprocessing():
    """
    Checks that dataset can be preprocessed
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10', 'torchvision', split=["test"])
    data.preprocess(224, 32)
    preprocessing_inputs = {'image_size': 224, 'batch_size': 32}
    assert data._preprocessed == preprocessing_inputs

@pytest.mark.pytorch
def test_defined_split():
    """
    Checks that dataset can be loaded into train and test subsets based on torchvision splits and then
    re-partitioned with shuffle-split
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10',
                       'torchvision', split=['train', 'test'])
    assert len(data.dataset) == 60000
    assert len(data.train_subset) == 50000
    assert len(data.test_subset) == 10000
    assert data.validation_subset is None
    assert data._train_indices == range(50000)
    assert data._test_indices == range(50000, 60000)
    assert data._validation_type == 'defined_split'

    # Apply shuffle split and verify new subset sizes
    data.shuffle_split(.6, .2, .2, seed=10)
    assert len(data.train_subset) == 36000
    assert len(data.validation_subset) == 12000
    assert len(data.test_subset) == 12000
    assert data._validation_type == 'shuffle_split'

@pytest.mark.pytorch
def test_shuffle_split():
    """
    Checks that dataset can be split into train, validation, and test subsets
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10', 'torchvision')
    data.shuffle_split(seed=10)
    assert len(data.train_subset) == 37500
    assert len(data.validation_subset) == 12500
    assert data.test_subset is None
    assert data._validation_type == 'shuffle_split'

@pytest.mark.pytorch
def test_shuffle_split_errors():
    """
    Checks that splitting into train, validation, and test subsets will error if inputs are wrong
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'GTSRB', 'torchvision')
    with pytest.raises(Exception) as e:
        data.shuffle_split(train_pct=.5, val_pct=.5, test_pct=.2)
    assert 'Sum of percentage arguments must be less than or equal to 1.' == str(e.value)

    with pytest.raises(Exception) as e:
        data.shuffle_split(train_pct=1, val_pct=0)
    assert 'Percentage arguments must be floats.' == str(e.value)

@pytest.mark.pytorch
def test_shuffle_split_deterministic():
    """
    Checks that dataset can be split into train, validation, and test subsets in a way that is reproducible
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'GTSRB', 'torchvision', split=['test'])
    data.preprocess(224, 128)
    data.shuffle_split(seed=10)

    data2 = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'GTSRB', 'torchvision', split=['test'])
    data2.preprocess(224, 128)
    data2.shuffle_split(seed=10)

    for i in range(3):
        image_1, label_1 = data.get_batch()
        image_2, label_2 = data2.get_batch()
        assert_array_equal(image_1, image_2)
        assert_array_equal(label_1, label_2)

@pytest.mark.pytorch
def test_batching():
    """
    Checks that dataset can be batched and then re-batched to a different batch size
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'GTSRB', 'torchvision')
    data.preprocess(224, 1)
    assert len(data.get_batch()[0]) == 1
    data.preprocess(224, 32)
    assert len(data.get_batch()[0]) == 32

@pytest.mark.pytorch
def test_batching_error():
    """
    Checks that preprocessing cannot be run twice with two different image_sizes
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'GTSRB', 'torchvision')
    data.preprocess(224, 1)
    with pytest.raises(Exception) as e:
        data.preprocess(256, 32)
    assert 'Data has already been preprocessed with a different image size: {}'.format(data._preprocessed) == \
           str(e.value)
