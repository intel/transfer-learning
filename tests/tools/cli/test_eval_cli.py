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
from pathlib import Path
from unittest.mock import MagicMock, patch
from tlt.tools.cli.commands.eval import eval
from tlt.utils.types import FrameworkType


@pytest.mark.common
@pytest.mark.parametrize('model_name,framework',
                         [['efficientnet_b0', FrameworkType.TENSORFLOW],
                          ['resnet50', FrameworkType.PYTORCH]])
@patch("tlt.models.model_factory.get_model")
@patch("tlt.datasets.dataset_factory.load_dataset")
@patch("inspect.getfullargspec")
def test_eval_preprocess_with_image_size(mock_inspect, mock_load_dataset, mock_get_model, model_name, framework):
    """
    Tests the eval command with a dataset preprocessing method that has an image_size. Actual calls for the model and
    dataset are mocked out. The test verifies that the proper args are used for calling preprocess()
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(tmp_dir, 'data')
    model_dir = os.path.join(tmp_dir, 'model')
    dummy_image_size = 100

    try:
        for new_dir in [model_dir, dataset_dir]:
            os.makedirs(new_dir)

        # Create dummy model file
        if framework == FrameworkType.TENSORFLOW:
            Path(os.path.join(model_dir, 'saved_model.pb')).touch()
        elif framework == FrameworkType.PYTORCH:
            Path(os.path.join(model_dir, 'model.pt')).touch()

        model_mock = MagicMock()
        model_mock.image_size = dummy_image_size
        data_mock = MagicMock()

        # Test where the preprocessing command will have an image size
        inspect_mock = MagicMock()
        inspect_mock.args = ['image_size', 'batch_size']
        mock_inspect.return_value = inspect_mock
        mock_get_model.return_value = model_mock
        mock_load_dataset.return_value = data_mock

        # Call the eval command
        result = runner.invoke(eval, ["--model-dir", model_dir, "--dataset_dir", dataset_dir])

        # Verify that the expected calls were made
        mock_load_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework)
        assert mock_get_model.called
        assert data_mock.shuffle_split.called
        assert model_mock.evaluate.called

        # Verify that preprocess was called with an image size
        data_mock.preprocess.assert_called_once_with(image_size=dummy_image_size, batch_size=32)

        # Verify that the eval command exit code is successful
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
def test_eval_preprocess_without_image_size(mock_inspect, mock_load_dataset, mock_get_model, model_name, framework):
    """
    Tests the eval command with a dataset preprocessing method that just has a batch size arg. Actual calls for the
    model and dataset are mocked out. The test verifies that the proper args are used for calling preprocess()
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(tmp_dir, 'data')
    model_dir = os.path.join(tmp_dir, 'model')
    dummy_image_size = 100

    try:
        for new_dir in [model_dir, dataset_dir]:
            os.makedirs(new_dir)

        # Create dummy model file
        if framework == FrameworkType.TENSORFLOW:
            Path(os.path.join(model_dir, 'saved_model.pb')).touch()
        elif framework == FrameworkType.PYTORCH:
            Path(os.path.join(model_dir, 'model.pt')).touch()

        model_mock = MagicMock()
        model_mock.image_size = dummy_image_size
        data_mock = MagicMock()

        # Test where the preprocessing command just has a batch_size arg
        inspect_mock = MagicMock()
        inspect_mock.args = ['batch_size']

        mock_inspect.return_value = inspect_mock
        mock_get_model.return_value = model_mock
        mock_load_dataset.return_value = data_mock

        # Call the eval command
        result = runner.invoke(eval, ["--model-dir", model_dir, "--dataset_dir", dataset_dir])

        # Verify that the eval command exit code is successful
        assert result.exit_code == 0

        # Verify that the expected calls were made
        mock_load_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework)
        assert mock_get_model.called
        assert data_mock.shuffle_split.called
        assert model_mock.evaluate.called

        # Verify that preprocess was called with just batch size
        data_mock.preprocess.assert_called_once_with(batch_size=32)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
@pytest.mark.parametrize('provided_model_name,model_dir,expected_model_name,framework',
                         [['mymodel/name', 'model/abc', 'mymodel/name', FrameworkType.TENSORFLOW],
                          ['', 'bert_en_uncased_L-12_H-768_A-12/3', 'bert_en_uncased_L-12_H-768_A-12',
                           FrameworkType.PYTORCH],
                          ['test', 'bert_en_uncased_L-12_H-768_A-12/3', 'test',
                           FrameworkType.PYTORCH]
                          ])
@patch("tlt.models.model_factory.get_model")
@patch("tlt.datasets.dataset_factory.load_dataset")
def test_eval_model_name(mock_load_dataset, mock_get_model, provided_model_name, model_dir,
                         expected_model_name, framework):
    """
    Tests the eval command with and without providing a model name to verify that when a model name is provided, that
    is what's used, and when a model name is not provided, we use the model_dir folder as the model name.
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(tmp_dir, 'data')
    model_dir = os.path.join(tmp_dir, model_dir)

    try:
        for new_dir in [model_dir, dataset_dir]:
            os.makedirs(new_dir)

        # Create dummy model file
        if framework == FrameworkType.TENSORFLOW:
            Path(os.path.join(model_dir, 'saved_model.pb')).touch()
        elif framework == FrameworkType.PYTORCH:
            Path(os.path.join(model_dir, 'model.pt')).touch()

        model_mock = MagicMock()
        data_mock = MagicMock()

        mock_get_model.return_value = model_mock
        mock_load_dataset.return_value = data_mock

        # Call the eval command
        eval_params = ["--model-dir", model_dir, "--dataset_dir", dataset_dir]

        if provided_model_name:
            eval_params += ["--model-name", provided_model_name]
        result = runner.invoke(eval, eval_params)

        # Verify that the expected calls were made
        mock_get_model.assert_called_once_with(expected_model_name, framework)
        mock_load_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework)
        assert model_mock.evaluate.called

        # Verify that the eval command exit code is successful
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
def test_eval_dataset_catalog(mock_get_dataset, mock_get_model, model_name, framework, dataset_name, dataset_catalog):
    """
    Tests the eval command a named dataset and verifies that get_dataset is called (vs load_dataset, which is used
    for custom dataset directories in other tests).
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(tmp_dir, 'data')
    model_dir = os.path.join(tmp_dir, 'model')

    try:
        for new_dir in [model_dir, dataset_dir]:
            os.makedirs(new_dir)

        # Create dummy model file
        if framework == FrameworkType.TENSORFLOW:
            Path(os.path.join(model_dir, 'saved_model.pb')).touch()
        elif framework == FrameworkType.PYTORCH:
            Path(os.path.join(model_dir, 'model.pt')).touch()

        # Setup mocks
        model_mock = MagicMock()
        data_mock = MagicMock()
        mock_get_model.return_value = model_mock
        mock_get_dataset.return_value = data_mock

        # Call the eval command
        result = runner.invoke(eval,
                               ["--model-dir", str(model_dir), "--model-name", model_name, "--dataset_dir", dataset_dir,
                                "--dataset-name", dataset_name, "--dataset-catalog", dataset_catalog])

        # Verify that the expected calls were made
        mock_get_model.assert_called_once_with(model_name, framework)
        mock_get_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework,
                                                 dataset_name, dataset_catalog)

        # Verify that the evaluate command exit code is successful
        assert model_mock.evaluate.called
        assert result.exit_code == 0
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


class TestEvalArgs:
    """
    Class for tests that are testing bad inputs for evaluation
    """

    def setup_class(self):
        self._runner = CliRunner()

        self._tmp_dir = tempfile.mkdtemp()
        self._dataset_dir = os.path.join(self._tmp_dir, 'data')
        self._model_dir = os.path.join(self._tmp_dir, 'model')

    def setup_method(self):
        for new_dir in [self._model_dir, self._dataset_dir]:
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

        Path(os.path.join(self._model_dir, 'saved_model.pb')).touch()

    def teardown_method(self):
        if os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)

    def teardown_class(self):
        if os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)

    @pytest.mark.common
    @pytest.mark.parametrize('dataset_catalog', ['foo', 'benchmark', '0'])
    def test_eval_invalid_dataset_catalog(self, dataset_catalog):
        """
        Verifies that eval command fails if the dataset catalog value is invalid
        """

        result = self._runner.invoke(eval,
                                     ["--model-dir", self._model_dir,
                                      "--dataset-dir", self._dataset_dir,
                                      "--dataset-name", "foo",
                                      "--dataset-catalog", dataset_catalog])

        assert result.exit_code == 2
        assert "Invalid value for '--dataset-catalog'" in result.output
        assert "'{}' is not one of 'tf_datasets', 'torchvision', 'huggingface'".format(dataset_catalog) in result.output
