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
from tlk.datasets.image_classification.tf_image_classification_dataset import TFImageClassificationDataset

# TODO: Create a fixture to initialize dataset and delete it once all tests have run
def test_tf_flowers():
    """
    Checks that a tf_flowers dataset can be created/loaded
    """
    flowers = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers', 'tf_datasets')
    assert type(flowers) == TFImageClassificationDataset
    assert len(flowers.class_names) == 5
    assert len(flowers.dataset) == 3670

def test_tf_flowers_10pct():
    """
    Checks that a 10% tf_flowers subset can be loaded
    """
    flowers = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers',
                                          'tf_datasets', split=["train[:10%]"])
    assert type(flowers) == TFImageClassificationDataset
    assert len(flowers.dataset) < 3670

def test_preprocessing():
    """
    Checks that dataset can be preprocessed only once
    """
    flowers = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers',
                                          'tf_datasets', split=["train[:10%]"])
    flowers.preprocess(224, 32)
    preprocessing_inputs = {'image_size': 224, 'batch_size': 32}
    assert flowers._preprocessed == preprocessing_inputs
    # Trying to preprocess again should throw an exception
    with pytest.raises(Exception) as e:
        assert 'Data has already been preprocessed: {}'.format(preprocessing_inputs) == e
    print(flowers.info)

def test_defined_split():
    """
    Checks that dataset can be loaded into train, validation, and test subsets based on TFDS splits and then
    re-partitioned with shuffle-split
    """
    beans = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'beans',
                          'tf_datasets', split=["train", "validation"])
    assert len(beans.dataset) == 1167
    assert len(beans.train_subset) == 1034
    assert len(beans.validation_subset) == 133
    assert beans.test_subset is None
    assert beans._validation_type == 'defined_split'

    # Apply shuffle split and verify new subset sizes
    beans.shuffle_split(.6, .2, .2, seed=10)
    assert len(beans.train_subset) == 700
    assert len(beans.validation_subset) == 233
    assert len(beans.test_subset) == 234
    assert beans._validation_type == 'shuffle_split'

def test_shuffle_split():
    """
    Checks that dataset can be split into train, validation, and test subsets
    """
    flowers = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers',
                          'tf_datasets', split=["train[:30%]"])
    flowers.shuffle_split(seed=10)
    assert len(flowers.train_subset) == 825
    assert len(flowers.validation_subset) == 275
    assert flowers.test_subset is None
    assert flowers._validation_type == 'shuffle_split'

def test_shuffle_split_errors():
    """
    Checks that splitting into train, validation, and test subsets will error if inputs are wrong
    """
    flowers = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers', 'tf_datasets')
    with pytest.raises(Exception) as e:
        flowers.shuffle_split(train_pct=.5, val_pct=.5, test_pct=.2)
    assert 'Sum of percentage arguments must be less than or equal to 1.' == str(e.value)
    with pytest.raises(Exception) as e:
        flowers.shuffle_split(train_pct=1, val_pct=0)
    assert 'Percentage arguments must be floats.' == str(e.value)

def test_shuffle_split_deterministic():
    """
    Checks that dataset can be split into train, validation, and test subsets in a way that is reproducible
    """
    flowers = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers', 'tf_datasets')
    flowers.preprocess(224, 1)
    flowers.shuffle_split(seed=10)

    flowers2 = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers', 'tf_datasets')
    flowers2.preprocess(224, 1)
    flowers2.shuffle_split(seed=10)

    for i in range(10):
        image_1, label_1 = flowers.get_batch()
        image_2, label_2 = flowers2.get_batch()
        assert_array_equal(image_1, image_2)
        assert_array_equal(label_1, label_2)

def test_batching():
    """
    Checks that dataset can be batched and then re-batched to a different batch size
    """
    flowers = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers', 'tf_datasets')
    flowers.preprocess(224, 1)
    assert len(flowers.get_batch()[0]) == 1
    flowers.preprocess(224, 32)
    assert  len(flowers.get_batch()[0]) == 32

def test_batching_error():
    """
    Checks that preprocessing cannot be run twice with two different image_sizes
    """
    flowers = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers', 'tf_datasets')
    flowers.preprocess(224, 1)
    with pytest.raises(Exception) as e:
        flowers.preprocess(256, 32)
    assert 'Data has already been preprocessed with a different image size: {}'.format(flowers._preprocessed) == \
           str(e.value)
