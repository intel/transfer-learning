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

from tlk.models import model_factory
from tlk.models.image_classification.tfhub_image_classification_model import TFHubImageClassificationModel
from tlk.utils.types import FrameworkType, UseCaseType


def test_tfhub_efficientnet_b0():
    """
    Checks that an efficientnet_b0 model can be downloaded from TFHub
    """
    model = model_factory.get_model('efficientnet_b0', 'tensorflow')
    assert type(model) == TFHubImageClassificationModel
    assert model.image_size == 224


def test_get_supported_models():
    """
    Call get supported models and checks to make sure the dictionary has keys for each use case,
    and checks for a known supported model.
    """
    model_dict = model_factory.get_supported_models()

    # Ensure there are keys for each use case
    for k in UseCaseType:
        assert str(k) in model_dict.keys()

    # Check for a known model
    assert 'efficientnet_b0' in model_dict[str(UseCaseType.IMAGE_CLASSIFICATION)]
    efficientnet_b0 = model_dict[str(UseCaseType.IMAGE_CLASSIFICATION)]['efficientnet_b0']
    assert str(FrameworkType.TENSORFLOW) in efficientnet_b0
    assert 'TFHub' == efficientnet_b0[str(FrameworkType.TENSORFLOW)]['model_hub']


@pytest.mark.parametrize('framework,use_case',
                         [['tensorflow', None],
                          ['pytorch', None],
                          [None, 'image_classification'],
                          [None, 'question_answering'],
                          ['tensorflow', 'image_classification'],
                          ['pytorch', 'text_classification'],
                          ['pytorch', 'question_answering']])
def test_get_supported_models_with_filter(framework, use_case):
    """
    Tests getting the dictionary of supported models while filtering by framework and/or use case.
    Checks to ensure that keys for the expected use cases are there. If filtering by framework, then the test will
    also check to make sure we only have models for the specified framework.
    """
    model_dict = model_factory.get_supported_models(framework, use_case)

    if use_case is not None:
        # Model dictionary should only have a key for the specified use case
        assert 1 == len(model_dict.keys())
        assert use_case in model_dict
    else:
        # Model dictionary should have keys for every use case
        assert len(UseCaseType) == len(model_dict.keys())
        for k in UseCaseType:
            assert str(k) in model_dict.keys()

    # If filtering by framework, we should not find models from other frameworks
    if framework is not None:
        for use_case_key in model_dict.keys():
            for model_name_key in model_dict[use_case_key].keys():
                assert 1 == len(model_dict[use_case_key][model_name_key].keys())
                assert framework in model_dict[use_case_key][model_name_key]


@pytest.mark.parametrize('bad_framework',
                         ['tensorflowers',
                          'python',
                          'torch',
                          'fantastic-potato'])
def test_get_supported_models_bad_framework(bad_framework):
    """
    Ensure that the proper error is raised when a bad framework is passed in
    """
    with pytest.raises(ValueError) as e:
        model_factory.get_supported_models(bad_framework)
        assert "Unsupported framework: {}".format(bad_framework) in str(e)


@pytest.mark.parametrize('bad_use_case',
                         ['tensorflow',
                          'imageclassification',
                          'python',
                          'fantastic-potato'])
def test_get_supported_models_bad_use_case(bad_use_case):
    """
    Ensure that the proper error is raised when a bad use case is passed in
    """
    with pytest.raises(ValueError) as e:
        model_factory.get_supported_models(use_case=bad_use_case)
        assert "Unsupported use case: {}".format(bad_use_case) in str(e)
