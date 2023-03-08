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
from tlt.tools.cli.commands.benchmark import benchmark
from tlt.utils.types import FrameworkType


@pytest.mark.common
@pytest.mark.parametrize('model_name,framework,batch_size,mode',
                         [['efficientnet_b0', FrameworkType.TENSORFLOW, 512, 'performance'],
                          ['inception_v3', FrameworkType.TENSORFLOW, 32, 'accuracy'],
                          ['resnet50', FrameworkType.PYTORCH, 128, 'performance'],
                          ['efficientnet_b2', FrameworkType.PYTORCH, 256, 'accuracy']])
@patch("tlt.models.model_factory.get_model")
@patch("tlt.datasets.dataset_factory.load_dataset")
def test_benchmark(mock_load_dataset, mock_get_model, model_name, framework, batch_size, mode):
    """
    Tests the benchmark comamnd with an without an Intel Neural Compressor config file and verifies that the
    expected calls are made on the tlt model object. The call parameters also verify that the benchmark command
    is able to properly identify the model's name based on the directory and the framework type based on the
    type of saved model.
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, model_name, '3')
    dataset_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        for new_dir in [model_dir, dataset_dir]:
            os.makedirs(new_dir)

        if framework == FrameworkType.TENSORFLOW:
            Path(os.path.join(model_dir, 'saved_model.pb')).touch()
        elif framework == FrameworkType.PYTORCH:
            Path(os.path.join(model_dir, 'model.pt')).touch()

        model_mock = MagicMock()
        data_mock = MagicMock()

        mock_get_model.return_value = model_mock
        mock_load_dataset.return_value = data_mock

        # Call the benchmark command without an Intel Neural Compressor config file
        result = runner.invoke(benchmark,
                               ["--model-dir", model_dir, "--dataset_dir", dataset_dir, "--mode", mode,
                                "--batch-size", batch_size, "--output-dir", output_dir])

        # Verify that the expected calls were made, including to create an Intel Neural Compressor config file
        mock_get_model.assert_called_once_with(model_name, framework)
        mock_load_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework)
        assert model_mock.write_inc_config_file.called
        assert model_mock.benchmark.called

        # Verify a successful exit code
        assert result.exit_code == 0

        # Reset mocks to do another experiment with an Intel Neural Compressor confijg file
        model_mock.reset_mock()
        data_mock.reset_mock()
        mock_get_model.reset_mock()
        mock_load_dataset.reset_mock()

        # Create a temp inc config yaml file
        inc_config = os.path.join(tmp_dir, 'inc_config.yaml')
        Path(inc_config).touch()

        # Call benchmark with a config file
        result = runner.invoke(benchmark,
                               ["--model-dir", model_dir, "--dataset_dir", dataset_dir, "--inc-config", inc_config])

        mock_get_model.assert_called_once_with(model_name, framework)
        mock_load_dataset.assert_called_once_with(dataset_dir, model_mock.use_case, model_mock.framework)
        model_mock.benchmark.called_once_with(model_dir, inc_config, mode)

        # Function to create an Intel Neural Compressor config file shouldn't have been called, since yaml was provided
        model_mock.write_inc_config_file.assert_not_called()

        # Verify a successful exit code
        assert result.exit_code == 0
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
@pytest.mark.parametrize('model_name,model_file',
                         [['bar', 'unsupported_model_type.txt'],
                          ['foo', 'potato.pb']])
def test_benchmark_bad_model_file(model_name, model_file):
    """
    Verifies that the benchmark command fails if it's given a model directory that doesn't contain a saved_model.pb or
    model.pt file.
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, model_name, '3')
    dataset_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        for new_dir in [model_dir, dataset_dir]:
            os.makedirs(new_dir)

        # Create the bogus model file
        Path(os.path.join(model_dir, model_file)).touch()

        # Call the benchmark command with the bogus model directory
        result = runner.invoke(benchmark,
                               ["--model-dir", model_dir, "--dataset_dir", dataset_dir, "--output-dir",
                                output_dir])

        # Verify that we got an error about the unsupported model type
        assert result.exit_code == 1
        assert "Benchmarking is currently only implemented for TensorFlow saved_model.pb and PyTorch model.pt models." \
               in result.output
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
@pytest.mark.parametrize('model_name,model_file,framework',
                         [['bar', 'saved_model.pb', 'tensorflow'],
                          ['foo', 'model.pt', 'pytorch']])
def test_benchmark_bad_model_dir(model_name, model_file, framework):
    """
    Verifies that benchmark command fails if it's given a model directory with a model name that we don't support
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, model_name, '3')
    dataset_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        for new_dir in [model_dir, dataset_dir]:
            os.makedirs(new_dir)

        # Create the model file
        Path(os.path.join(model_dir, model_file)).touch()

        # Call the benchmark command with the model directory
        result = runner.invoke(benchmark,
                               ["--model-dir", model_dir, "--dataset_dir", dataset_dir, "--output-dir", output_dir])

        # Verify that we got an error about the unsupported model for the framework
        assert result.exit_code == 1
        assert "An error occurred while getting the model" in result.output
        assert "The specified model is not supported for {}".format(framework) in result.output
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
def test_benchmark_model_dir_does_not_exist():
    """
    Verifies that benchmark command fails if the model directory does not exist
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, 'resnet_v1_50', '3')
    dataset_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        os.makedirs(dataset_dir)

        # Call the benchmark command with the model directory
        result = runner.invoke(benchmark,
                               ["--model-dir", model_dir, "--dataset_dir", dataset_dir, "--output-dir", output_dir])

        # Verify that we got an error model directory not existing
        assert result.exit_code == 2
        assert "--model-dir" in result.output
        assert "Directory '{}' does not exist".format(model_dir) in result.output
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
def test_benchmark_dataset_dir_does_not_exist():
    """
    Verifies that benchmark command fails if the dataset directory does not exist
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, 'resnet_v1_50', '3')
    dataset_dir = os.path.join(tmp_dir, 'data')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        os.makedirs(model_dir)

        # Call the benchmark command with the model directory
        result = runner.invoke(benchmark,
                               ["--model-dir", model_dir, "--dataset_dir", dataset_dir, "--output-dir", output_dir])

        # Verify that we got an error dataset directory not existing
        assert result.exit_code == 2
        assert "--dataset-dir" in result.output
        assert "Directory '{}' does not exist".format(dataset_dir) in result.output
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


class TestBenchmarkArgs:
    """
    Class for tests that are testing bad inputs for benchmarking args with generic folders for the model dir,
    dataset dir, and output dir.
    """

    def setup_class(self):
        self._runner = CliRunner()

        self._tmp_dir = tempfile.mkdtemp()
        self._model_dir = os.path.join(self._tmp_dir, 'resnet_v1_50', '3')
        self._dataset_dir = os.path.join(self._tmp_dir, 'data')
        self._output_dir = os.path.join(self._tmp_dir, 'output')

    def setup_method(self):
        for new_dir in [self._model_dir, self._dataset_dir]:
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

    def teardown_method(self):
        if os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)

    def teardown_class(self):
        if os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)

    @pytest.mark.common
    @pytest.mark.parametrize('batch_size',
                             ['foo', 'benchmark', '0', -1, 0])
    def test_benchmark_invalid_batch_size(self, batch_size):
        """
        Verifies that benchmark command fails if the batch size is invalid
        """
        # Create the model file
        Path(os.path.join(self._model_dir, 'saved_model.pt')).touch()

        # Call the benchmark command with the model directory
        result = self._runner.invoke(benchmark,
                                     ["--model-dir", self._model_dir,
                                      "--dataset_dir", self._dataset_dir,
                                      "--output-dir", self._output_dir,
                                      "--batch-size", batch_size])

        assert result.exit_code == 2
        assert "Invalid value for '--batch-size'" in result.output

    @pytest.mark.common
    @pytest.mark.parametrize('mode',
                             ['foo', 'benchmark', '0'])
    def test_benchmark_invalid_mode(self, mode):
        """
        Verifies that benchmark command fails if the mode value is invalid (the choices are: accuracy, performance)
        """

        # Create the model file
        Path(os.path.join(self._model_dir, 'saved_model.pt')).touch()

        # Call the benchmark command with the model directory
        result = self._runner.invoke(benchmark,
                                     ["--model-dir", self._model_dir,
                                      "--dataset_dir", self._dataset_dir,
                                      "--output-dir", self._output_dir,
                                      "--mode", mode])

        assert result.exit_code == 2
        assert "Invalid value for '--mode'" in result.output
        assert "'{}' is not one of 'performance', 'accuracy'".format(mode) in result.output
