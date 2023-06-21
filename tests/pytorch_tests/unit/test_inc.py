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
import uuid

from pathlib import Path
from unittest.mock import patch, MagicMock

from tlt.models import model_factory
from tlt.datasets.image_classification.pytorch_custom_image_classification_dataset import PyTorchCustomImageClassificationDataset  # noqa: E501


try:
    # Do PyTorch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlt.models.image_classification.torchvision_image_classification_model import TorchvisionImageClassificationModel  # noqa: F401, E501
except ModuleNotFoundError:
    print("WARNING: Unable to import TorchvisionImageClassificationModel. PyTorch or torchvision may not be installed")


@pytest.mark.pytorch
def test_torchvision_image_classification_optimize_graph_not_implemented():
    """
    Verifies the error that gets raise if graph optimization is attempted with a PyTorch model
    """
    try:
        output_dir = tempfile.mkdtemp()
        saved_model_dir = tempfile.mkdtemp()
        dummy_config_file = os.path.join(output_dir, "config.yaml")
        Path(dummy_config_file).touch()
        model = model_factory.get_model('resnet50', 'pytorch')
        # The torchvision model is not present until training, so call _get_hub_model()
        model._get_hub_model(3)
        # Graph optimization is not enabled for PyTorch, so this should fail
        with patch('neural_compressor.experimental.Graph_Optimization'):
            with pytest.raises(NotImplementedError):
                model.optimize_graph(output_dir)

        # Verify that the installed version of Intel Neural Compressor throws a SystemError
        from neural_compressor.experimental import Graph_Optimization, common
        # set_backend API is no longer available in Neural Compressor v2.0
        # from neural_compressor.experimental.common.model import set_backend
        # set_backend('pytorch')
        graph_optimizer = Graph_Optimization()
        with pytest.raises(AssertionError):
            graph_optimizer.model = common.Model(model._model)

    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)


@pytest.mark.pytorch
@patch('tlt.models.image_classification.torchvision_image_classification_model.ModelDownloader')
@patch('tlt.models.pytorch_model.quantization.fit')
def test_pyt_image_classification_quantize_overwrite_saved_model(mock_quantization_fit, mock_model_downloader):
    """
    Given a valid directory for the output dir, test the quantize function with the actual Intel Neural
    Compressor call mocked out. Tests that the model will be overwritten or not using the overwrite_model flag.
    """

    # tlt imports
    from tlt.datasets.image_classification.pytorch_custom_image_classification_dataset \
        import PyTorchCustomImageClassificationDataset
    from tlt.models import model_factory

    try:
        # Specify a directory for output
        output_dir = tempfile.mkdtemp()

        model = model_factory.get_model(model_name='efficientnet_b0', framework='pytorch')

        # Mock the dataset
        mock_dataset = MagicMock()
        mock_dataset.__class__ = PyTorchCustomImageClassificationDataset
        mock_dataset.get_inc_dataloaders.return_value = 1, 2

        # Method to create a dummy model.pt file in the specified directory
        def create_dummy_file(output_dir):
            with open(os.path.join(output_dir, 'model.pt'), 'w') as _:
                pass

        # Mock an INC quantized model that will create a dummy file when saved
        mock_quantized_model = MagicMock()
        mock_quantized_model.save.side_effect = create_dummy_file

        # Mock the INC quantization.fit method
        def mock_fit(**args):
            return mock_quantized_model
        mock_quantization_fit.side_effect = mock_fit

        # Call quantize when a model does not exist
        model.quantize(output_dir=output_dir, dataset=mock_dataset, overwrite_model=False)

        # Call quantize when the model exists, but overwrite_model=True
        model.quantize(output_dir=output_dir, dataset=mock_dataset, overwrite_model=True)
        model.quantize(output_dir=output_dir, dataset=mock_dataset, overwrite_model=True)

        with pytest.raises(FileExistsError):  # Model exists, so this should be true
            model.quantize(output_dir=output_dir, dataset=mock_dataset, overwrite_model=False)

    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@pytest.mark.pytorch
def test_pyt_image_classification_quantization():
    """
    Given a valid directory for output dir, test the quantization function with the actual INC called mocked out.
    """
    try:
        output_dir = tempfile.mkdtemp()
        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            with patch('neural_compressor.quantization.fit') as mock_q:
                mock_dataset.dataset_dir = "/tmp/data/my_photos"
                mock_dataset.__class__ = PyTorchCustomImageClassificationDataset
                mock_dataset.get_inc_dataloaders.return_value = (1, 2)
                model.quantize(output_dir, mock_dataset)
                mock_q.assert_called_once()
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@pytest.mark.pytorch
def test_pyt_image_classification_benchmark_model_does_not_exist():
    """
    Verifies the error that gets raise if benchmarking is done with a model that does not exist
    """
    try:
        output_dir = tempfile.mkdtemp()
        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            mock_dataset.dataset_dir = "/tmp/data/my_photos"
            mock_dataset.__class__ = PyTorchCustomImageClassificationDataset
            random_dir = str(uuid.uuid4())
            saved_model_dir = tempfile.mkdtemp()

            with patch('neural_compressor.experimental.Benchmark'):
                # It's not a directory, so we expect an error
                with pytest.raises(NotADirectoryError):
                    model.benchmark(mock_dataset, saved_model_dir=random_dir)

                # An empty directory with no saved model should also generate an error
                with pytest.raises(FileNotFoundError):
                    model.benchmark(mock_dataset, saved_model_dir=saved_model_dir)

    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)


@pytest.mark.pytorch
def test_pyt_image_classification_inc_benchmark():
    """
    Verifies that if we have a valid model and dataset, benchmarking is called. The actual benchmarking calls to INC
    are mocked out.
    """
    model = model_factory.get_model('efficientnet_b0', 'pytorch')
    with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
        with patch('neural_compressor.benchmark.fit') as mock_bench:
            mock_dataset.dataset_dir = "/tmp/data/my_photos"
            mock_dataset.__class__ = PyTorchCustomImageClassificationDataset
            mock_dataset.get_inc_dataloaders.return_value = (1, 2)
            model.benchmark(mock_dataset)
            mock_bench.assert_called_once()
