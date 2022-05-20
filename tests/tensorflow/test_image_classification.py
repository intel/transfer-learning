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

import os
import pytest

from tlk.datasets import dataset_factory
from tlk.models import model_factory


@pytest.mark.parametrize('model_name,dataset_name',
                         [['efficientnet_b0', 'tf_flowers'],
                          ['resnet_v1_50', 'tf_flowers']])
def test_tf_image_classification(model_name, dataset_name):
    """
    Tests basic transfer learning functionality for TensorFlow image classification models using TF Datasets
    """
    framework = 'tensorflow'
    output_dir = '/tmp/output'

    # Get the dataset
    dataset = dataset_factory.get_dataset('/tmp/data', 'image_classification', framework, dataset_name,
                                          'tf_datasets', split=["train[:5%]"])

    # Get the model
    model = model_factory.get_model(model_name, framework)

    # Preprocess the dataset and train
    dataset.preprocess(model.image_size, 32)
    dataset.shuffle_split(seed=10)
    model.train(dataset, output_dir=output_dir, epochs=1)

    # Evaluate
    model.evaluate(dataset)

    # Predict with a batch
    images, labels = dataset.get_batch()
    predictions = model.predict(images)
    assert len(predictions) == 32

    # export the saved model
    saved_model_dir = model.export(output_dir)
    assert os.path.isdir(saved_model_dir)
    assert os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb"))

    # Reload the saved model
    reload_model = model_factory.get_model(model_name, framework)
    reload_model.load_from_directory(saved_model_dir)

    # Evaluate
    metrics = reload_model.evaluate(dataset)
    assert len(metrics) > 0
