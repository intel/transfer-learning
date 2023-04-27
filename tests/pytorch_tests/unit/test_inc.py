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
from unittest.mock import patch

from tlt.models import model_factory

try:
    # Do PyTorch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlt.models.image_classification.torchvision_image_classification_model import TorchvisionImageClassificationModel  # noqa: F401, E501
except ModuleNotFoundError:
    print("WARNING: Unable to import TorchvisionImageClassificationModel. PyTorch or torchvision may not be installed")

from tlt.datasets import dataset_factory
from tlt.utils.file_utils import download_and_extract_tar_file
from tlt.datasets.image_classification.pytorch_custom_image_classification_dataset import PyTorchCustomImageClassificationDataset  # noqa: E501

# Load a custom PyTorch dataset that can be re-used for tests
dataset_dir = tempfile.mkdtemp()
custom_dataset_path = os.path.join(dataset_dir, "flower_photos")
if not os.path.exists(custom_dataset_path):
    download_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    download_and_extract_tar_file(download_url, dataset_dir)
# Load the dataset from the custom dataset path
dataset = dataset_factory.load_dataset(dataset_dir=custom_dataset_path,
                                       use_case='image_classification',
                                       framework='pytorch')


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
                model.optimize_graph(saved_model_dir, output_dir)

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
def test_pyt_image_classification_config_file_overwrite():
    """
    Tests writing an INC config file for image classification models with a mock custom dataset. Checks that the
    overwrite flag lets you overwrite a config file that already exists.
    """
    try:
        temp_dir = tempfile.mkdtemp()
        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            mock_dataset.__class__ = PyTorchCustomImageClassificationDataset
            config_file = os.path.join(temp_dir, "config.yaml")
            batch_size = 4
            mock_dataset.dataset_dir = "/tmp/data/my_photos"
            nc_workspace = os.path.join(temp_dir, "nc_workspace")
            model.write_inc_config_file(config_file, dataset, batch_size=batch_size, tuning_workspace=nc_workspace)
            assert os.path.exists(config_file)

            # If overwrite=False this should fail, since the config file already exists
            with pytest.raises(FileExistsError):
                model.write_inc_config_file(config_file, dataset, batch_size=batch_size, overwrite=False)

            # Writing the config file again should work with overwrite=True
            model.write_inc_config_file(config_file, dataset, batch_size=batch_size, overwrite=True)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.pytorch
@pytest.mark.parametrize('batch_size,valid',
                         [[1, True],
                          [-1, False],
                          ['abc', False],
                          [1.434, False],
                          [0, False],
                          [128, True]])
def test_pyt_image_classification_config_file_batch_size(batch_size, valid):
    """
    Tests writing an INC config file with good and bad batch sizes
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, dataset, batch_size=batch_size, overwrite=True,
                                                tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, dataset, batch_size=batch_size, overwrite=True,
                                            tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.pytorch
@pytest.mark.parametrize('resize_interpolation,valid',
                         [['bilinear', True],
                          [-1, False],
                          ['nearest', True],
                          [1.434, False],
                          ['bicubic', True],
                          ['foo', False]])
def test_pyt_image_classification_config_file_resize_interpolation(resize_interpolation, valid):
    """
    Tests writing an INC config file with good and bad resize_interpolation values
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, dataset, batch_size=1, overwrite=True,
                                                resize_interpolation=resize_interpolation,
                                                tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, dataset, batch_size=1, overwrite=True,
                                            resize_interpolation=resize_interpolation, tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.pytorch
@pytest.mark.parametrize('accuracy_criterion,valid',
                         [[0.1, True],
                          [-1, False],
                          [0.01, True],
                          [1.434, False],
                          ['foo', False]])
def test_pyt_image_classification_config_file_accuracy_criterion(accuracy_criterion, valid):
    """
    Tests writing an INC config file with good and bad accuracy_criterion_relative values
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, dataset, batch_size=1, overwrite=True,
                                                accuracy_criterion_relative=accuracy_criterion,
                                                tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, dataset, batch_size=1, overwrite=True,
                                            accuracy_criterion_relative=accuracy_criterion,
                                            tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.pytorch
@pytest.mark.parametrize('timeout,valid',
                         [[0.1, False],
                          [-1, False],
                          [0, True],
                          [60, True],
                          ['foo', False]])
def test_pyt_image_classification_config_file_timeout(timeout, valid):
    """
    Tests writing an INC config file with good and bad exit_policy_timeout values
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, dataset, batch_size=1, overwrite=True,
                                                exit_policy_timeout=timeout, tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, dataset, batch_size=1, overwrite=True,
                                            exit_policy_timeout=timeout, tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.pytorch
@pytest.mark.parametrize('max_trials,valid',
                         [[0.1, False],
                          [-1, False],
                          [0, False],
                          [1, True],
                          [60, True],
                          ['foo', False]])
def test_pyt_image_classification_config_file_max_trials(max_trials, valid):
    """
    Tests writing an INC config file with good and bad exit_policy_max_trials values
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, dataset, batch_size=1, overwrite=True,
                                                exit_policy_max_trials=max_trials, tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, dataset, batch_size=1, overwrite=True,
                                            exit_policy_max_trials=max_trials, tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.pytorch
@pytest.mark.parametrize('seed,valid',
                         [[0.1, False],
                          [-1, False],
                          [0, True],
                          [1, True],
                          [123, True],
                          ['foo', False]])
def test_pyt_image_classification_config_file_seed(seed, valid):
    """
    Tests writing an INC config file with good and bad tuning_random_seed values
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, dataset, batch_size=1, overwrite=True,
                                                tuning_random_seed=seed, tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, dataset, batch_size=1, overwrite=True,
                                            tuning_random_seed=seed, tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.pytorch
def test_pyt_image_classification_quantization():
    """
    Given valid directories for the saved model, output dir, and config file, test the quantization function with
    the actual INC called mocked out.
    """
    try:
        output_dir = tempfile.mkdtemp()
        saved_model_dir = tempfile.mkdtemp()
        saved_model_file = os.path.join(saved_model_dir, "model.pt")
        Path(saved_model_file).touch()
        dummy_config_file = os.path.join(saved_model_dir, "config.yaml")
        Path(dummy_config_file).touch()

        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            with patch('neural_compressor.experimental.Quantization') as mock_q:
                mock_dataset.dataset_dir = "/tmp/data/my_photos"

                model.quantize(saved_model_dir, output_dir, dummy_config_file)
                mock_q.assert_called_with(dummy_config_file)
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)


@pytest.mark.pytorch
def test_pyt_image_classification_quantization_model_does_not_exist():
    """
    Verifies the error that gets raise if quantization or INC benchmarking is done with a model that does not exist
    """
    try:
        output_dir = tempfile.mkdtemp()
        dummy_config_file = os.path.join(output_dir, "config.yaml")
        Path(dummy_config_file).touch()
        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            mock_dataset.dataset_dir = "/tmp/data/my_photos"
            with patch('neural_compressor.experimental.Quantization'):

                # Generate a random name that wouldn't exist
                random_dir = str(uuid.uuid4())

                # It's not a directory, so we expect an error
                with pytest.raises(NotADirectoryError):
                    model.quantize(random_dir, output_dir, dummy_config_file)

                saved_model_dir = tempfile.mkdtemp()

                # An empty directory with no saved model should also generate an error
                with pytest.raises(FileNotFoundError):
                    model.quantize(saved_model_dir, output_dir, dummy_config_file)

            with patch('neural_compressor.experimental.Benchmark'):
                # It's not a directory, so we expect an error
                with pytest.raises(NotADirectoryError):
                    model.benchmark(random_dir, dummy_config_file, model_type='int8')

                # An empty directory with no saved model should also generate an error
                with pytest.raises(FileNotFoundError):
                    model.benchmark(saved_model_dir, dummy_config_file, model_type='int8')

    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)


def test_pyt_image_classification_inc_benchmark():
    """
    Verifies that if we have valid parameters for the saved model, config file, and mode, benchmarking is called. The
    actual benchmarking calls to INC are mocked out.
    """
    try:
        output_dir = tempfile.mkdtemp()
        saved_model_dir = tempfile.mkdtemp()
        saved_model_file = os.path.join(saved_model_dir, "model.pt")
        Path(saved_model_file).touch()
        dummy_config_file = os.path.join(saved_model_dir, "config.yaml")
        Path(dummy_config_file).touch()

        model = model_factory.get_model('efficientnet_b0', 'pytorch')
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            with patch('neural_compressor.experimental.Benchmark') as mock_bench:
                mock_dataset.dataset_dir = "/tmp/data/my_photos"

                model.benchmark(saved_model_dir, dummy_config_file)
                mock_bench.assert_called_with(dummy_config_file)
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)


@pytest.mark.pytorch
@pytest.mark.parametrize('mode,valid',
                         [['abc', False],
                          [1, False],
                          [0, False],
                          ['performance', True],
                          ['accuracy', True]])
def test_pyt_image_classification_inc_benchmark_mode(mode, valid):
    """
    Checks error handling for the benchmarking mode
    """
    output_dir = tempfile.mkdtemp()
    model = model_factory.get_model(model_name='efficientnet_b0', framework='pytorch')
    dataset = dataset_factory.get_dataset(dataset_dir=dataset_dir,
                                          use_case='image_classification',
                                          framework='pytorch',
                                          dataset_name='CIFAR10',
                                          dataset_catalog='torchvision')
    batch_size = 456
    dataset.preprocess(model.image_size, batch_size=batch_size)
    dataset.shuffle_split(train_pct=0.05, val_pct=0.05, shuffle_files=False)
    model.train(dataset, output_dir=output_dir, epochs=1)
    try:
        saved_model_dir = tempfile.mkdtemp()
        dummy_config_file = os.path.join(saved_model_dir, "config.yaml")
        Path(dummy_config_file).touch()
        import dill
        import torch
        model_copy = dill.dumps(model._model)
        torch.save(model_copy, os.path.join(saved_model_dir, 'model.pt'))
        with patch('tlt.datasets.image_classification.pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset') as mock_dataset:  # noqa: E501
            with patch('neural_compressor.experimental.Benchmark') as mock_bench:
                mock_dataset.dataset_dir = "/tmp/data/my_photos"

                if not valid:
                    with pytest.raises(ValueError):
                        model.benchmark(saved_model_dir, dummy_config_file, mode=mode)
                else:
                    model.benchmark(saved_model_dir, dummy_config_file, mode=mode)
                    mock_bench.assert_called_with(dummy_config_file)
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
