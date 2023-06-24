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

from unittest.mock import patch, MagicMock
from tlt.models import model_factory

try:
    # Do TF specific imports in a try/except to prevent pytest test loading from failing when running in a PyTorch env
    from tlt.models.image_classification.tf_image_classification_model import TFImageClassificationModel  # noqa: F401
    from tlt.models.image_classification.tf_image_classification_model import TFCustomImageClassificationDataset
except ModuleNotFoundError:
    print("WARNING: Unable to import TFImageClassificationModel. TensorFlow may not be installed")


@pytest.mark.tensorflow
def test_tf_image_classification_quantization():
    """
    Given a valid directory for the output dir, test the quantization function with the actual Intel Neural Compressor
    call mocked out.
    """
    try:
        output_dir = tempfile.mkdtemp()

        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            with patch('neural_compressor.quantization.fit') as mock_q:
                mock_dataset.dataset_dir = "/tmp/data/my_photos"
                mock_dataset.__class__ = TFCustomImageClassificationDataset
                mock_dataset.get_inc_dataloaders.return_value = (1, 2)
                model.quantize(output_dir, mock_dataset)
                mock_q.assert_called_once()
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@pytest.mark.tensorflow
@patch('tlt.models.tf_model.quantization.fit')
def test_tf_image_classification_quantize_overwrite_saved_model(mock_quantization_fit):
    """
    Given a valid directory for the output dir, test the quantize function with the actual Intel Neural
    Compressor call mocked out. Tests that the model will be overwritten or not using the overwrite_model flag.
    """

    from tlt.models import model_factory

    try:
        # Specify a directory for output
        output_dir = tempfile.mkdtemp()

        model = model_factory.get_model(model_name='resnet_v1_50', framework='tensorflow')

        # Mock the dataset
        mock_dataset = MagicMock()
        mock_dataset.__class__ = TFCustomImageClassificationDataset
        mock_dataset.get_inc_dataloaders.return_value = 1, 2

        # Method to create a dummy model.pt file in the specified directory
        def create_dummy_file(output_dir):
            with open(os.path.join(output_dir, 'saved_model.pb'), 'w') as fp:
                fp.close()

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


@patch('tlt.models.tf_model.Graph_Optimization')
@pytest.mark.tensorflow
def test_tf_image_classification_optimize_graph_overwrite_saved_model(mock_graph_optimization):
    """
    Given a valid directory for the output dir, test the quantize function with the actual Intel Neural
    Compressor call mocked out. Tests that the model will be overwritten or not using the overwrite_model flag.
    """

    # tlt imports
    from tlt.models.image_classification.tf_image_classification_model import TFCustomImageClassificationDataset
    from tlt.models import model_factory

    try:
        # Specify a directory for output
        output_dir = tempfile.mkdtemp()

        model = model_factory.get_model(model_name='resnet_v1_50', framework='tensorflow')

        # Mock the dataset
        mock_dataset = MagicMock()
        mock_dataset.__class__ = TFCustomImageClassificationDataset
        mock_dataset.get_inc_dataloaders.return_value = 1, 2

        # Method to create a dummy model.pt file in the specified directory
        def create_dummy_file():
            with open(os.path.join(output_dir, 'saved_model.pb'), 'w') as fp:
                fp.close()
            return MagicMock()

        # Mock an INC quantized model that will create a dummy file when saved
        mock_graph_optimization.side_effect = create_dummy_file

        # Call optimize_graph when a model does not exist
        model.optimize_graph(output_dir=output_dir)

        # Call optimize_graph when the model exists, but overwrite_model=True
        model.optimize_graph(output_dir=output_dir, overwrite_model=True)
        model.optimize_graph(output_dir=output_dir, overwrite_model=True)

        with pytest.raises(FileExistsError):  # Model exists, so this should be true
            model.optimize_graph(output_dir=output_dir, overwrite_model=False)

    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@pytest.mark.tensorflow
def test_tf_image_classification_benchmark_model_does_not_exist():
    """
    Verifies the error that gets raise if benchmarking is done with a model that does not exist
    """
    try:
        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            mock_dataset.dataset_dir = "/tmp/data/my_photos"
            mock_dataset.__class__ = TFCustomImageClassificationDataset
            random_dir = str(uuid.uuid4())
            saved_model_dir = tempfile.mkdtemp()
            with patch('neural_compressor.benchmark.fit'):
                # It's not a directory, so we expect an error
                with pytest.raises(NotADirectoryError):
                    model.benchmark(mock_dataset, saved_model_dir=random_dir)

                # An empty directory with no saved model should also generate an error
                with pytest.raises(FileNotFoundError):
                    model.benchmark(mock_dataset, saved_model_dir=saved_model_dir)
    finally:
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)


@pytest.mark.tensorflow
def test_tf_image_classification_inc_benchmark():
    """
    Verifies that if we have a valid model and dataset, benchmarking is called. The actual benchmarking calls to Intel
    Neural Compressor are mocked out.
    """
    model = model_factory.get_model('efficientnet_b0', 'tensorflow')
    with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
            as mock_dataset:
        with patch('neural_compressor.benchmark.fit') as mock_bench:
            mock_dataset.dataset_dir = "/tmp/data/my_photos"
            mock_dataset.__class__ = TFCustomImageClassificationDataset
            mock_dataset.get_inc_dataloaders.return_value = (1, 2)
            model.benchmark(mock_dataset)
            mock_bench.assert_called_once()
