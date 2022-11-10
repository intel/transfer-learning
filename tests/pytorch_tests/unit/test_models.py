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

from unittest.mock import MagicMock, patch

from tlt.models import model_factory
from tlt.utils.types import FrameworkType, UseCaseType

try:
    # Do torch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    import torch
except ModuleNotFoundError as e:
    print("WARNING: Unable to import torch. Torch may not be installed")


try:
    # Do torch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlt.models.image_classification.torchvision_image_classification_model import TorchvisionImageClassificationModel
    from tlt.datasets.image_classification.torchvision_image_classification_dataset import TorchvisionImageClassificationDataset
except ModuleNotFoundError as e:
    print("WARNING: Unable to import TorchvisionImageClassificationModel and TorchvisionImageClassificationDataset. "
          "Torch may not be installed")

try:
    from tlt.datasets.text_classification.hf_text_classification_dataset import HFTextClassificationDataset
except ModuleNotFoundError:
    print("WARNING: Unable to import HFTextClassificationDataset. HuggingFace's `tranformers` API may not be installed \
           in the current env")


@pytest.mark.pytorch
def test_torchvision_efficientnet_b0():
    """
    Checks that an efficientnet_b0 model can be downloaded from TFHub
    """
    model = model_factory.get_model('efficientnet_b0', 'pytorch')
    assert type(model) == TorchvisionImageClassificationModel
    assert model.model_name == 'efficientnet_b0'


@pytest.mark.pytorch
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
    assert str(FrameworkType.PYTORCH) in efficientnet_b0
    assert 'torchvision' == efficientnet_b0[str(FrameworkType.PYTORCH)]['model_hub']


@pytest.mark.pytorch
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


@pytest.mark.pytorch
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


@pytest.mark.pytorch
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


@pytest.mark.pytorch
def test_torchvision_efficientnet_b0_train():
    """
    Tests calling train on a torchvision efficientnet_b0 model with a mock dataset, model, and optimizer
    """
    model = model_factory.get_model('efficientnet_b0', 'pytorch')
    model._generate_checkpoints = False

    with patch('tlt.datasets.image_classification.torchvision_image_classification_dataset.TorchvisionImageClassificationDataset') \
            as mock_dataset:
        with patch('tlt.models.image_classification.torchvision_image_classification_model.'
                   'TorchvisionImageClassificationModel._get_hub_model') as mock_get_hub_model:
            mock_dataset.train_subset = [1, 2, 3]
            mock_dataset.validation_subset = [4, 5, 6]
            mock_dataset.__class__ = TorchvisionImageClassificationDataset
            mock_model = MagicMock()
            mock_optimizer = MagicMock()
            expected_return_value_model = mock_model
            expected_return_value_history_val = {'Acc': [0.0], 'Loss': [0.0], 'Val Acc': [0.0], 'Val Loss': [0.0]}
            expected_return_value_history_no_val = {'Acc': [0.0], 'Loss': [0.0]}

            def mock_to(device):
                assert device == torch.device("cpu")
                return expected_return_value_model

            def mock_train():
                return None

            mock_model.to = mock_to
            mock_model.train = mock_train
            mock_get_hub_model.return_value = (mock_model, mock_optimizer)

            # Train and eval (eval should be called)
            return_val = model.train(mock_dataset, output_dir="/tmp/output/pytorch", do_eval=True, lr_decay=False)
            assert return_val == expected_return_value_history_val
            mock_model.eval.assert_called_once()

            # Train without eval (eval should not be called)
            mock_model.eval.reset_mock()
            return_val = model.train(mock_dataset, output_dir="/tmp/output/pytorch", do_eval=False, lr_decay=False)
            assert return_val == expected_return_value_history_no_val
            mock_model.eval.assert_not_called()

            # Try to train with eval, but no validation subset (eval should not be called)
            mock_dataset.validation_subset = None
            mock_model.eval.reset_mock()
            return_val = model.train(mock_dataset, output_dir="/tmp/output/pytorch", do_eval=True, lr_decay=False)
            assert return_val == expected_return_value_history_no_val
            mock_model.eval.assert_not_called()
