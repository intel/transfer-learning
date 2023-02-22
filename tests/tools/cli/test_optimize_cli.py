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
from tlt.tools.cli.commands.optimize import optimize
from tlt.utils.types import FrameworkType


@pytest.mark.common
@pytest.mark.parametrize('model_name,framework',
                         [['efficientnet_b0', FrameworkType.TENSORFLOW],
                          ['inception_v3', FrameworkType.TENSORFLOW],
                          ['resnet50', FrameworkType.PYTORCH]])
@patch("tlt.models.model_factory.get_model")
def test_optimize(mock_get_model, model_name, framework):
    """
    Tests the optimize commandand verifies that the expected calls are made
    on the tlt model object. The call parameters also verify that the optimize command is able to properly identify
    the model's name based on the directory and the framework type based on the type of saved model.
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, model_name, '3')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        os.makedirs(model_dir)

        if framework == FrameworkType.TENSORFLOW:
            Path(os.path.join(model_dir, 'saved_model.pb')).touch()
        elif framework == FrameworkType.PYTORCH:
            Path(os.path.join(model_dir, 'model.pt')).touch()

        model_mock = MagicMock()
        mock_get_model.return_value = model_mock

        # Call the optimize command
        result = runner.invoke(optimize,
                               ["--model-dir", model_dir, "--output-dir", output_dir])

        # Verify that the expected calls were made
        if framework == FrameworkType.TENSORFLOW:
            mock_get_model.assert_called_once_with(model_name, framework)
            assert model_mock.optimize_graph.called

        # Verify the exit code
        if framework == FrameworkType.TENSORFLOW:
            assert result.exit_code == 0
        else:
            assert result.exit_code == 1

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
@pytest.mark.parametrize('model_name,model_file',
                         [['bar', 'unsupported_model_type.txt'],
                          ['foo', 'potato.pb'],
                          ['pytorch_model', 'model.pt']])
def test_optimize_bad_model_file(model_name, model_file):
    """
    Verifies that the optimize command fails if it's given a model directory that doesn't contain a saved_model.pb.
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, model_name, '3')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        os.makedirs(model_dir)

        # Create the bogus model file
        Path(os.path.join(model_dir, model_file)).touch()

        # Call the optimize command with the bogus model directory
        result = runner.invoke(optimize,
                               ["--model-dir", model_dir, "--output-dir", output_dir])

        # Verify that we got an error about the unsupported model type
        assert result.exit_code == 1
        assert "Graph optimization is currently only supported for TensorFlow saved_model.pb models." \
               in result.output
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
@pytest.mark.parametrize('model_name,model_file,framework',
                         [['bar', 'saved_model.pb', 'tensorflow'],
                          ['foo', 'saved_model.pb', 'tensorflow']])
def test_optimize_bad_model_dir(model_name, model_file, framework):
    """
    Verifies that optimize command fails if it's given a model directory with a model name that we don't support
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, model_name, '3')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        os.makedirs(model_dir)

        # Create the model file
        Path(os.path.join(model_dir, model_file)).touch()

        # Call the optimize command with the model directory
        result = runner.invoke(optimize,
                               ["--model-dir", model_dir, "--output-dir", output_dir])

        # Verify that we got an error about the unsupported model for the framework
        assert result.exit_code == 1
        assert "An error occurred while getting the model" in result.output
        assert "The specified model is not supported for {}".format(framework) in result.output
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
def test_optimize_model_dir_does_not_exist():
    """
    Verifies that optimize command fails if the model directory does not exist
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, 'resnet_v1_50', '3')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        # Call the optimize command with the model directory
        result = runner.invoke(optimize,
                               ["--model-dir", model_dir, "--output-dir", output_dir])

        # Verify that we got an error model directory not existing
        assert result.exit_code == 2
        assert "--model-dir" in result.output
        assert "Directory '{}' does not exist".format(model_dir) in result.output
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.common
@patch("tlt.models.model_factory.get_model")
def test_optimize_output_dir(mock_get_model):
    """
    Verifies that the optimize command increments the output directory for the optimized model each time
    the optimization command is called
    """
    runner = CliRunner()

    tmp_dir = tempfile.mkdtemp()
    model_name = 'resnet_v1_50'
    model_dir = os.path.join(tmp_dir, model_name, '3')
    output_dir = os.path.join(tmp_dir, 'output')

    try:
        os.makedirs(model_dir)
        Path(os.path.join(model_dir, 'saved_model.pb')).touch()

        model_mock = MagicMock()
        mock_get_model.return_value = model_mock

        for i in range(1, 5):
            # Call the optimize command
            result = runner.invoke(optimize,
                                   ["--model-dir", model_dir, "--output-dir", output_dir])
            assert result.exit_code == 0

            # Check for an expected optimization output dir with the folder number incrementing
            expected_optimize_dir = os.path.join(output_dir, "optimize", model_name, str(i))
            model_mock.optimize.called_once_with(model_dir, expected_optimize_dir)

            model_mock.reset_mock()

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
