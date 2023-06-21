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

import os
import pytest
import shutil
import tempfile

from click.testing import CliRunner
from unittest.mock import MagicMock, patch
from tlt.tools.cli.commands.train import train
from tlt.utils.types import FrameworkType


@pytest.mark.common
@pytest.mark.parametrize('model_name,framework',
                         [['efficientnet_b0', FrameworkType.TENSORFLOW],
                          ['resnet50', FrameworkType.PYTORCH]])
@patch("tlt.models.model_factory.get_model")
@patch("tlt.datasets.dataset_factory.load_dataset")
@patch("inspect.getfullargspec")
def test_train_preprocess_with_image_size(mock_inspect, mock_load_dataset, mock_get_model, model_name, framework):
    """
    Tests the train command with a dataset preprocessing method that has an image_size. Actual calls for the model and
    dataset are mocked out. The test verifies that the proper args are used for calling preprocess()
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')
    dummy_image_size = 100

    try:
        for new_dir in [output_dir, dataset_dir]:
            os.makedirs(new_dir)

        model_mock = MagicMock()
        model_mock.image_size = dummy_image_size
        data_mock = MagicMock()

        # Test where the preprocessing command will have an image size
        inspect_mock = MagicMock()
        inspect_mock.args = ['image_size', 'batch_size']
        mock_inspect.return_value = inspect_mock
        mock_get_model.return_value = model_mock
        mock_load_dataset.return_value = data_mock
        model_mock.export.return_value = output_dir

        # Call the train command
        result = runner.invoke(train,
                               ["--framework", str(framework), "--model-name", model_name, "--dataset_dir", dataset_dir,
                                "--output-dir", output_dir])

        # Verify that the expected calls were made
        mock_get_model.assert_called_once_with(model_name, str(framework))
        mock_load_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework)
        assert data_mock.shuffle_split.called
        assert model_mock.train.called

        # Verify that preprocess was called with the right arguments
        data_mock.preprocess.assert_called_once_with(image_size=dummy_image_size, batch_size=32, add_aug=[])

        # Verify that the train command exit code is successful
        assert result.exit_code == 0
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
@pytest.mark.parametrize('model_name,framework',
                         [['google/bert_uncased_L-10_H-128_A-2', FrameworkType.TENSORFLOW],
                          ['bert_en_uncased_L-12_H-768_A-12', FrameworkType.PYTORCH]])
@patch("tlt.models.model_factory.get_model")
@patch("tlt.datasets.dataset_factory.load_dataset")
@patch("inspect.getfullargspec")
def test_train_preprocess_without_image_size(mock_inspect, mock_load_dataset, mock_get_model, model_name, framework):
    """
    Tests the train command with a dataset preprocessing method that just has a batch size arg. Actual calls for the
    model and dataset are mocked out. The test verifies that the proper args are used for calling preprocess()
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')
    dummy_image_size = 100

    try:
        for new_dir in [output_dir, dataset_dir]:
            os.makedirs(new_dir)

        model_mock = MagicMock()
        model_mock.image_size = dummy_image_size
        data_mock = MagicMock()

        # Test where the preprocessing command just has a batch_size arg
        inspect_mock = MagicMock()
        inspect_mock.args = ['batch_size']

        mock_inspect.return_value = inspect_mock
        mock_get_model.return_value = model_mock
        mock_load_dataset.return_value = data_mock
        model_mock.export.return_value = output_dir

        # Call the train command
        result = runner.invoke(train,
                               ["--framework", str(framework), "--model-name", model_name, "--dataset_dir", dataset_dir,
                                "--output-dir", output_dir])

        # Verify that the expected calls were made
        mock_get_model.assert_called_once_with(model_name, str(framework))
        mock_load_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework)
        assert data_mock.shuffle_split.called
        assert model_mock.train.called

        # Verify preprocess was called with the right arguments
        data_mock.preprocess.assert_called_once_with(batch_size=32)

        # Verify that the train command exit code is successful
        assert result.exit_code == 0
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
@pytest.mark.parametrize('model_name,framework,add_aug',
                         [['efficientnet_b0', FrameworkType.TENSORFLOW, 'rotate'],
                          ['resnet50', FrameworkType.PYTORCH, 'zoom']])
@patch("tlt.models.model_factory.get_model")
@patch("tlt.datasets.dataset_factory.load_dataset")
@patch("inspect.getfullargspec")
def test_train_add_augmentation(mock_inspect, mock_load_dataset, mock_get_model, model_name, framework, add_aug):
    """
    Tests the train command with add augmentation. Actual calls for the model and dataset are mocked out. The test
    verifies that the proper args are passed to the model train() method.
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        for new_dir in [output_dir, dataset_dir]:
            os.makedirs(new_dir)

        model_mock = MagicMock()
        data_mock = MagicMock()
        mock_get_model.return_value = model_mock
        mock_load_dataset.return_value = data_mock
        model_mock.export.return_value = output_dir

        # Call the train command
        result = runner.invoke(train,
                               ["--framework", str(framework), "--model-name", model_name, "--dataset_dir", dataset_dir,
                                "--output-dir", output_dir, "--add_aug", add_aug])

        # Verify that the expected calls were made
        mock_get_model.assert_called_once_with(model_name, str(framework))
        mock_load_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework)
        assert data_mock.shuffle_split.called
        assert model_mock.train.called

        # Verify preprocess was called with the right arguments
        data_mock.preprocess.assert_called_once_with(batch_size=32)

        # Verify that the train command exit code is successful
        assert result.exit_code == 0
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
@pytest.mark.parametrize('model_name,framework,init_checkpoints',
                         [['bert_en_uncased_L-12_H-768_A-12', FrameworkType.TENSORFLOW, '/tmp/checkpoints'],
                          ['resnet50', FrameworkType.PYTORCH, '/tmp/checkpoint.pt']])
@patch("tlt.models.model_factory.get_model")
@patch("tlt.datasets.dataset_factory.load_dataset")
def test_train_init_checkpoints(mock_load_dataset, mock_get_model, model_name, framework, init_checkpoints):
    """
    Tests the train command with init checkpoints. Actual calls for the model and dataset are mocked out. The test
    verifies that the proper args are passed to the model train() method.
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        for new_dir in [output_dir, dataset_dir]:
            os.makedirs(new_dir)

        # Setup mocks
        model_mock = MagicMock()
        data_mock = MagicMock()
        mock_get_model.return_value = model_mock
        mock_load_dataset.return_value = data_mock
        model_mock.export.return_value = output_dir

        # Call the train command
        result = runner.invoke(train,
                               ["--framework", str(framework), "--model-name", model_name, "--dataset_dir", dataset_dir,
                                "--output-dir", output_dir, "--init-checkpoints", init_checkpoints, "--epochs", 2])

        # Verify that the expected calls were made
        mock_get_model.assert_called_once_with(model_name, str(framework))
        mock_load_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework)

        # Verify that train and preprocess were called with the right arguments
        if framework == FrameworkType.TENSORFLOW:
            model_mock.train.assert_called_once_with(data_mock, output_dir=output_dir, epochs=2,
                                                     initial_checkpoints=init_checkpoints, early_stopping=False,
                                                     lr_decay=False, distributed=False, hostfile=None, nnodes=1,
                                                     nproc_per_node=1, use_horovod=False)
        elif framework == FrameworkType.PYTORCH:
            model_mock.train.assert_called_once_with(data_mock, output_dir=output_dir, epochs=2,
                                                     initial_checkpoints=init_checkpoints, early_stopping=False,
                                                     lr_decay=False, ipex_optimize=False, distributed=False,
                                                     hostfile=None, nnodes=1, nproc_per_node=1)
        data_mock.preprocess.assert_called_once_with(batch_size=32)

        # Verify that the train command exit code is successful
        assert result.exit_code == 0
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
@pytest.mark.parametrize('model_name,framework,epochs,early_stopping, lr_decay',
                         [['efficientnet_b0', FrameworkType.TENSORFLOW, 15, True, False],
                          ['resnet50', FrameworkType.PYTORCH, 15, True, True],
                          ['efficientnet_b0', FrameworkType.TENSORFLOW, 15, False, True],
                          ['resnet50', FrameworkType.PYTORCH, 15, False, False]])
@patch("tlt.models.model_factory.get_model")
@patch("tlt.datasets.dataset_factory.load_dataset")
@patch("inspect.getfullargspec")
def test_train_features(mock_inspect, mock_load_dataset, mock_get_model, model_name, framework, epochs, early_stopping, lr_decay):  # noqa: E501
    """
    Tests the train command with early stopping. Actual calls for the model and dataset are mocked out. The test
    verifies that the proper args are passed to the model train() method.
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        for new_dir in [output_dir, dataset_dir]:
            os.makedirs(new_dir)

        model_mock = MagicMock()
        data_mock = MagicMock()
        mock_get_model.return_value = model_mock
        mock_load_dataset.return_value = data_mock
        model_mock.export.return_value = output_dir

        # Call the train command
        if early_stopping and lr_decay:
            result = runner.invoke(train,
                                   ["--framework", str(framework), "--model-name", model_name, "--dataset_dir",
                                    dataset_dir, "--output-dir", output_dir, "--epochs", epochs, "--early_stopping",
                                    "--lr_decay"])
        elif early_stopping:
            result = runner.invoke(train,
                                   ["--framework", str(framework), "--model-name", model_name, "--dataset_dir",
                                    dataset_dir, "--output-dir", output_dir, "--epochs", epochs, "--early_stopping"])
        elif lr_decay:
            result = runner.invoke(train,
                                   ["--framework", str(framework), "--model-name", model_name, "--dataset_dir",
                                    dataset_dir, "--output-dir", output_dir, "--epochs", epochs, "--lr_decay"])

        else:
            result = runner.invoke(train,
                                   ["--framework", str(framework), "--model-name", model_name, "--dataset_dir",
                                    dataset_dir, "--output-dir", output_dir, "--epochs", epochs])

        # Verify that the expected calls were made
        mock_get_model.assert_called_once_with(model_name, str(framework))
        mock_load_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework)
        assert data_mock.shuffle_split.called
        assert model_mock.train.called

        # Verify that train and preprocess were called with the right arguments
        if framework == FrameworkType.TENSORFLOW:
            model_mock.train.assert_called_once_with(data_mock, output_dir=output_dir, epochs=15,
                                                     initial_checkpoints=None, early_stopping=early_stopping,
                                                     lr_decay=lr_decay, distributed=False, hostfile=None, nnodes=1,
                                                     nproc_per_node=1, use_horovod=False)
        elif framework == FrameworkType.PYTORCH:
            model_mock.train.assert_called_once_with(data_mock, output_dir=output_dir, epochs=15,
                                                     initial_checkpoints=None, early_stopping=early_stopping,
                                                     lr_decay=lr_decay, ipex_optimize=False, distributed=False,
                                                     hostfile=None, nnodes=1, nproc_per_node=1)

        # Verify that the train command exit code is successful
        assert result.exit_code == 0
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
@pytest.mark.parametrize('model_name,framework,dataset_name,dataset_catalog',
                         [['efficientnet_b0', FrameworkType.TENSORFLOW, 'tf_flowers', 'tf_datasets'],
                          ['resnet50', FrameworkType.PYTORCH, 'cifar10', 'torchvision']])
@patch("tlt.models.model_factory.get_model")
@patch("tlt.datasets.dataset_factory.get_dataset")
def test_train_dataset_catalog(mock_get_dataset, mock_get_model, model_name, framework, dataset_name, dataset_catalog):
    """
    Tests the train command a named dataset and verifies that get_dataset is called (vs load_dataset, which is used
    for custom dataset directories in other tests).
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        for new_dir in [output_dir, dataset_dir]:
            os.makedirs(new_dir)

        # Setup mocks
        model_mock = MagicMock()
        data_mock = MagicMock()
        mock_get_model.return_value = model_mock
        mock_get_dataset.return_value = data_mock
        model_mock.export.return_value = output_dir

        # Call the train command
        result = runner.invoke(train,
                               ["--framework", str(framework), "--model-name", model_name, "--dataset_dir", dataset_dir,
                                "--output-dir", output_dir, "--dataset-name", dataset_name,
                                "--dataset-catalog", dataset_catalog])

        # Verify that the expected calls were made
        mock_get_model.assert_called_once_with(model_name, str(framework))
        mock_get_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework,
                                                 dataset_name, dataset_catalog)

        # Verify that the train command exit code is successful
        assert model_mock.train.called
        assert result.exit_code == 0
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


class TestTrainArgs:
    """
    Class for tests that are testing bad inputs for training with generic folders for the dataset dir and output dir.
    """

    def setup_class(self):
        self._runner = CliRunner()

        self._tmp_dir = tempfile.mkdtemp()
        self._dataset_dir = os.path.join(self._tmp_dir, 'data')
        self._output_dir = os.path.join(self._tmp_dir, 'output')

    def setup_method(self):
        for new_dir in [self._output_dir, self._dataset_dir]:
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

    def teardown_method(self):
        if os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)

    def teardown_class(self):
        if os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)

    @pytest.mark.common
    @pytest.mark.parametrize('epochs',
                             ['foo', 'benchmark', '0', -1, 0])
    def test_train_invalid_epochs(self, epochs):
        """
        Verifies that train command fails if the epoch parameter is invalid
        """

        result = self._runner.invoke(train,
                                     ["--model-name", "foo",
                                      "--dataset_dir", self._dataset_dir,
                                      "--output-dir", self._output_dir,
                                      "--epochs", epochs])

        assert result.exit_code == 2
        assert "Invalid value for '--epochs'" in result.output

    @pytest.mark.common
    @pytest.mark.parametrize('framework',
                             ['foo', 'benchmark', '0'])
    def test_train_invalid_framework(self, framework):
        """
        Verifies that train command fails if the framework value is invalid
        """

        result = self._runner.invoke(train,
                                     ["--model-name", "foo",
                                      "--dataset_dir", self._dataset_dir,
                                      "--output-dir", self._output_dir,
                                      "--framework", framework])

        assert result.exit_code == 2
        assert "Invalid value for '--framework'" in result.output
        assert "'{}' is not one of 'tensorflow', 'pytorch'".format(framework) in result.output

    @pytest.mark.common
    @pytest.mark.parametrize('dataset_catalog', ['foo', 'benchmark', '0'])
    def test_train_invalid_dataset_catalog(self, dataset_catalog):
        """
        Verifies that train command fails if the dataset catalog value is invalid
        """

        result = self._runner.invoke(train,
                                     ["--model-name", "foo",
                                      "--dataset_dir", self._dataset_dir,
                                      "--output-dir", self._output_dir,
                                      "--dataset-catalog", dataset_catalog])

        assert result.exit_code == 2
        assert "Invalid value for '--dataset-catalog'" in result.output
        assert "'{}' is not one of 'tf_datasets', 'torchvision', 'huggingface'".format(dataset_catalog) in result.output
