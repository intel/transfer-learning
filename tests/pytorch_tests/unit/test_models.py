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
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy

from unittest import mock
from unittest.mock import ANY, MagicMock, patch
from sklearn import decomposition

from tlt.models import model_factory
from tlt.utils.types import FrameworkType, UseCaseType

try:
    from tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model import extract_features, pca, get_feature_extraction_model  # noqa: E501
except ModuleNotFoundError:
    print("WARNING: Unable to import PytorchImageAnomolyDetectionModel. Pytorch may not be installed")

# This is necessary to protect from import errors when testing in a pytorch only environment
# True when imports are successful, False when imports are unsuccessful
torch_env = True

try:
    # Do torch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    print("WARNING: Unable to import torch. Torch may not be installed")
    torch_env = False

try:
    # Do torch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlt.models.image_classification.torchvision_image_classification_model import TorchvisionImageClassificationModel  # noqa: E501
    from tlt.datasets.image_classification.torchvision_image_classification_dataset import TorchvisionImageClassificationDataset   # noqa: E501
    from tlt.datasets.image_classification.pytorch_custom_image_classification_dataset import \
        PyTorchCustomImageClassificationDataset  # noqa: E501
    from tlt.models.text_classification.pytorch_hf_text_classification_model import PyTorchHFTextClassificationModel  # noqa: E501
except ModuleNotFoundError:
    print("WARNING: Unable to import TorchvisionImageClassificationModel and TorchvisionImageClassificationDataset. "
          "Torch may not be installed")

try:
    from tlt.models.image_anomaly_detection.torchvision_image_anomaly_detection_model import \
        TorchvisionImageAnomalyDetectionModel
except ModuleNotFoundError:
    print("WARNING: Unable to import TorchvisionImageAnomalyDetectionModel and "
          "PyTorchCustomImageAnomalyDetectionDataset. Torch may not be installed")

try:
    from tlt.datasets.text_classification.hf_text_classification_dataset import HFTextClassificationDataset  # noqa: F401, E501
except ModuleNotFoundError:
    print("WARNING: Unable to import HFTextClassificationDataset. Hugging Face's `transformers` API may not \
           be installed in the current env")


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
    assert 'resnet50' in model_dict[str(UseCaseType.IMAGE_ANOMALY_DETECTION)]
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
                          ['pytorch', 'question_answering'],
                          ['pytorch', 'image_anomaly_detection']])
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

    with patch('tlt.datasets.image_classification.torchvision_image_classification_dataset.TorchvisionImageClassificationDataset') as mock_dataset:  # noqa: E501
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


@pytest.mark.pytorch
def test_bert_train():
    model = model_factory.get_model('distilbert-base-uncased', 'pytorch')
    assert type(model) == PyTorchHFTextClassificationModel
    with patch('tlt.datasets.text_classification.hf_text_classification_dataset.HFTextClassificationDataset') as mock_dataset:  # noqa: E501
        mock_dataset.__class__ = HFTextClassificationDataset
        mock_dataset.train_subset = ['1', '2', '3']
        mock_dataset.validation_subset = ['4', '5', '6']
        expected_return_value_history_no_val = {'Acc': [0.0], 'Loss': [0.0]}
        expected_return_value_history_val = {'Acc': [0.0], 'Loss': [0.0], 'Val Acc': [0.0], 'Val Loss': [0.0]}

        # Scenario 1: Call train without validation
        return_val = model.train(mock_dataset, output_dir="/tmp/output/pytorch", do_eval=False, lr_decay=False)
        assert return_val['Acc'] == expected_return_value_history_no_val['Acc']
        assert return_val['Loss'] == expected_return_value_history_no_val['Loss']
        assert 'train_runtime' in return_val
        assert 'train_samples_per_second' in return_val
        assert 'Val Acc' not in return_val
        assert 'Val Loss' not in return_val

        # Scenario 2: Call train with validation
        mock_dataset.validation_loader.__class__ = HFTextClassificationDataset
        return_val = model.train(mock_dataset, output_dir="/tmp/output/pytorch", do_eval=True, lr_decay=False)
        assert return_val['Acc'] == expected_return_value_history_val['Acc']
        assert return_val['Loss'] == expected_return_value_history_val['Loss']
        assert return_val['Val Acc'] == expected_return_value_history_val['Val Acc']
        assert return_val['Val Loss'] == expected_return_value_history_val['Val Loss']
        assert 'train_runtime' in return_val
        assert 'train_samples_per_second' in return_val


@pytest.mark.pytorch
def test_resnet50_anomaly_extract_pca():
    model = model_factory.get_model(model_name="resnet50", framework="pytorch", use_case="anomaly_detection")
    assert type(model) == TorchvisionImageAnomalyDetectionModel

    # Call extract_features and PCA on 5 randomly generated images
    data = torch.rand(5, 3, 225, 225)  # NCHW
    resnet_model = get_feature_extraction_model(model._model, 'layer3')
    features = extract_features(resnet_model, data, layer_name='layer3', pooling=['avg', 2])
    assert isinstance(features, torch.Tensor)
    assert len(features) == 5

    data_mats_orig = torch.empty((features.shape[1], len(data))).to('cpu')

    # Skip the rest of the test if the tensor contains any NaNs, due to flaky behavior
    if not numpy.isnan(data_mats_orig).any():
        with torch.no_grad():
            components = pca(data_mats_orig, 0.97)
        assert type(components) == decomposition._pca.PCA
        assert components.n_components == 0.97


# This is necessary to protect from import errors when testing in a pytorch only environment
if torch_env:
    @pytest.mark.pytorch
    @pytest.mark.parametrize('model_name,use_case,dataset_type,optimizer,loss',
                             [['efficientnet_b0', 'image_classification', PyTorchCustomImageClassificationDataset,
                               torch.optim.Adam, torch.nn.L1Loss],
                              ['resnet18', 'image_classification', PyTorchCustomImageClassificationDataset,
                               torch.optim.AdamW, torch.nn.MSELoss],
                              ['custom', 'image_classification', PyTorchCustomImageClassificationDataset,
                               torch.optim.SGD, torch.nn.L1Loss],
                              ['distilbert-base-uncased', 'text_classification', HFTextClassificationDataset,
                               torch.optim.Adam, torch.nn.MSELoss]])
    def test_pytorch_optimizer_loss(model_name, use_case, dataset_type, optimizer, loss):
        """
        Tests initializing and training a model with configurable optimizers and loss functions
        """

        # Define a model
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 3)

            def forward(self, x):
                x = self.pool(nn.functional.relu(self.conv1(x)))
                x = self.pool(nn.functional.relu(self.conv2(x)))
                x = torch.flatten(x, 1)
                x = nn.functional.relu(self.fc1(x))
                x = nn.functional.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        net = Net()

        if model_name == 'custom':
            model = model_factory.load_model(model_name, net, 'pytorch', use_case, optimizer=optimizer, loss=loss)
        else:
            model = model_factory.get_model(model_name, 'pytorch', optimizer=optimizer, loss=loss)

        model._generate_checkpoints = False
        model._fit = MagicMock()
        assert model._optimizer_class == optimizer
        assert model._loss_class == loss
        assert type(model._loss) == loss

        mock_dataset = MagicMock()
        mock_dataset.__class__ = dataset_type
        mock_dataset.class_names = ['a', 'b', 'c']
        mock_dataset.train_subset = [1, 2, 3]
        mock_dataset.validation_subset = [4, 5, 6]

        # Train is called and optimizer and loss objects should match the input types
        model.train(mock_dataset, output_dir="/tmp/output/pytorch")
        assert model._optimizer_class == optimizer
        assert type(model._optimizer) == optimizer
        assert model._loss_class == loss
        assert type(model._loss) == loss


# This is necessary to protect from import errors when testing in a pytorch only environment
if torch_env:
    @pytest.mark.pytorch
    @pytest.mark.parametrize('model_name,optimizer',
                             [['efficientnet_b0', 1],
                              ['resnet18', 'foo'],
                              ['distilbert-base-uncased', torch.nn.MSELoss]])
    def test_pytorch_optimizer_wrong_type(model_name, optimizer):
        """
        Tests that an exception is thrown when the input optimizer is the wrong type
        """
        with pytest.raises(TypeError):
            model_factory.get_model(model_name, 'pytorch', optimizer=optimizer)


@pytest.mark.pytorch
@patch('tlt.models.text_classification.pytorch_hf_text_classification_model.torch.optim.AdamW')
@patch('tlt.models.text_classification.pytorch_hf_text_classification_model.Trainer')
@patch('tlt.models.text_classification.pytorch_hf_text_classification_model.ModelDownloader')
def test_pytorch_hf_text_classification_trainer_return_values(mock_downloader, mock_trainer, mock_optimizer):
    """
    Tests the PyTorch Text Classification model with the Hugging Face Trainer to verify that the value returned
    by Trainer.train() is returned by the model.train() method
    """

    model = model_factory.get_model(model_name='bert-base-cased', framework='pytorch')

    mock_dataset = MagicMock()
    mock_dataset.__class__ = HFTextClassificationDataset
    mock_dataset.class_names = ['a', 'b', 'c']
    mock_dataset.train_subset = [1, 2, 3]
    mock_dataset.validation_subset = [4, 5, 6]

    expected_value = "a"

    mock_trainer().train.return_value = expected_value

    return_val = model.train(mock_dataset, output_dir="/tmp", use_trainer=True, seed=10)
    assert mock_trainer().train.call_count == 1

    assert return_val == expected_value


@pytest.mark.pytorch
@patch('tlt.models.text_classification.pytorch_hf_text_classification_model.torch.optim.AdamW')
@patch('tlt.models.text_classification.pytorch_hf_text_classification_model.Trainer')
@patch('tlt.models.text_classification.pytorch_hf_text_classification_model.ModelDownloader')
def test_pytorch_hf_text_classification_trainer_without_val_subset(mock_downloader, mock_trainer, mock_optimizer):
    """
    Tests the PyTorch Text Classification model with the Hugging Face Trainer is able to run evaluation with a test
    subset when a validation subset does not exist.
    """

    model = model_factory.get_model(model_name='bert-base-cased', framework='pytorch')

    mock_dataset = MagicMock()
    mock_dataset.__class__ = HFTextClassificationDataset
    mock_dataset.class_names = ['a', 'b', 'c']
    mock_dataset.train_subset = [1, 2, 3]
    mock_dataset.test_subset = [4, 5, 6]
    type(mock_dataset).validation_subset = mock.PropertyMock(side_effect=ValueError)

    with pytest.raises(ValueError):
        mock_dataset.validation_subset

    model.train(mock_dataset, output_dir="/tmp", use_trainer=True, seed=10)
    mock_trainer.assert_called_with(model=model._model, args=ANY, train_dataset=[1, 2, 3], eval_dataset=[4, 5, 6],
                                    compute_metrics=ANY, tokenizer=ANY)
