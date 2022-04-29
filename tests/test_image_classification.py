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
from tlk.models import model_factory

def test_tf_image_classification():
    """
    Tests basic transfer learning functionality on tf_flowers with efficientnet_b0
    """
    flowers = dataset_factory.get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers',
                                          'tf_datasets', split=["train[:5%]"])
    model = model_factory.get_model('efficientnet_b0', 'tensorflow')
    flowers.preprocess(model.image_size, 32)
    model.train(flowers, 1)
    images, labels = flowers.get_batch()
    predictions = model.predict(images)
    assert len(predictions) == 32
