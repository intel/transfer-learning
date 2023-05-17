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
import yaml

from pathlib import Path
from unittest.mock import patch

from tlt.models import model_factory

try:
    # Do TF specific imports in a try/except to prevent pytest test loading from failing when running in a PyTorch env
    from tlt.models.image_classification.tf_image_classification_model import TFImageClassificationModel  # noqa: F401
except ModuleNotFoundError:
    print("WARNING: Unable to import TFImageClassificationModel. TensorFlow may not be installed")

from tlt.datasets import dataset_factory
from tlt.utils.file_utils import download_and_extract_tar_file

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


@pytest.mark.tensorflow
def test_tf_image_classification_config_file_overwrite():
    """
    Tests writing an Intel Neural Compressor config file for image classification models with a mock custom dataset.
    Checks that the overwrite flag lets you overwrite a config file that already exists.
    """
    try:
        temp_dir = tempfile.mkdtemp()
        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            config_file = os.path.join(temp_dir, "config.yaml")
            batch_size = 24
            mock_dataset.dataset_dir = "/tmp/data/my_photos"
            nc_workspace = os.path.join(temp_dir, "nc_workspace")
            model.write_inc_config_file(config_file, mock_dataset, batch_size=batch_size, tuning_workspace=nc_workspace)
            assert os.path.exists(config_file)

            # If overwrite=False this should fail, since the config file already exists
            with pytest.raises(FileExistsError):
                model.write_inc_config_file(config_file, mock_dataset, batch_size=batch_size, overwrite=False)

            # Writing the config file again should work with overwrite=True
            model.write_inc_config_file(config_file, mock_dataset, batch_size=batch_size, overwrite=True)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.tensorflow
@pytest.mark.parametrize('validation_folder,validation_type',
                         [['test', 'defined_split'],
                          ['validation', 'defined_split']])
@patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset')
def test_tf_image_classification_defined_split_inc_config(mock_dataset, validation_folder, validation_type):
    """
    Tests writing an Intel Neural Compressor config file for an image classification dataset that has a 'defined split'
    to verify that the data loader directory has the expected directory
    """
    try:
        temp_dir = tempfile.mkdtemp()
        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.os.path.exists') as mock_path_exists:
            config_file = os.path.join(temp_dir, "config.yaml")
            batch_size = 24
            dataset_dir = "/tmp/data/my_photos"
            mock_dataset.dataset_dir = dataset_dir
            mock_dataset._validation_type = validation_type
            nc_workspace = os.path.join(temp_dir, "nc_workspace")

            def mocked_path_exists(dir_path):
                if dir_path.startswith(dataset_dir):
                    return os.path.basename(dir_path) == validation_folder
                else:
                    return True

            mock_path_exists.side_effect = mocked_path_exists

            model.write_inc_config_file(config_file, mock_dataset, batch_size=batch_size, tuning_workspace=nc_workspace)
            assert os.path.exists(config_file)

            with open(config_file) as f:
                config_dict = yaml.safe_load(f)

            # Check that the config has the expected data loader path
            expected_path = os.path.join(dataset_dir, validation_folder)

            assert config_dict["quantization"]["calibration"]["dataloader"]["dataset"]["ImageFolder"]["root"] == \
                   expected_path
            assert config_dict["evaluation"]["accuracy"]["dataloader"]["dataset"]["ImageFolder"]["root"] == \
                   expected_path
            assert config_dict["evaluation"]["performance"]["dataloader"]["dataset"]["ImageFolder"]["root"] == \
                   expected_path

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.tensorflow
@pytest.mark.parametrize('batch_size,valid',
                         [[1, True],
                          [-1, False],
                          ['abc', False],
                          [1.434, False],
                          [0, False],
                          [128, True]])
def test_tf_image_classification_config_file_batch_size(batch_size, valid):
    """
    Tests writing an Intel Neural Compressor config file with good and bad batch sizes
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, mock_dataset, batch_size=batch_size, overwrite=True,
                                                tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, mock_dataset, batch_size=batch_size, overwrite=True,
                                            tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.tensorflow
@pytest.mark.parametrize('resize_interpolation,valid',
                         [['bilinear', True],
                          [-1, False],
                          ['nearest', True],
                          [1.434, False],
                          ['bicubic', True],
                          ['foo', False]])
def test_tf_image_classification_config_file_resize_interpolation(resize_interpolation, valid):
    """
    Tests writing an Intel Neural Compressor config file with good and bad resize_interpolation values
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, mock_dataset, batch_size=1, overwrite=True,
                                                resize_interpolation=resize_interpolation,
                                                tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, mock_dataset, batch_size=1, overwrite=True,
                                            resize_interpolation=resize_interpolation, tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.tensorflow
@pytest.mark.parametrize('accuracy_criterion,valid',
                         [[0.1, True],
                          [-1, False],
                          [0.01, True],
                          [1.434, False],
                          ['foo', False]])
def test_tf_image_classification_config_file_accuracy_criterion(accuracy_criterion, valid):
    """
    Tests writing an Intel Neural Compressor config file with good and bad accuracy_criterion_relative values
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, mock_dataset, batch_size=1, overwrite=True,
                                                accuracy_criterion_relative=accuracy_criterion,
                                                tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, mock_dataset, batch_size=1, overwrite=True,
                                            accuracy_criterion_relative=accuracy_criterion,
                                            tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.tensorflow
@pytest.mark.parametrize('timeout,valid',
                         [[0.1, False],
                          [-1, False],
                          [0, True],
                          [60, True],
                          ['foo', False]])
def test_tf_image_classification_config_file_timeout(timeout, valid):
    """
    Tests writing an Intel Neural Compressor config file with good and bad exit_policy_timeout values
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, mock_dataset, batch_size=1, overwrite=True,
                                                exit_policy_timeout=timeout, tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, mock_dataset, batch_size=1, overwrite=True,
                                            exit_policy_timeout=timeout, tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.tensorflow
@pytest.mark.parametrize('max_trials,valid',
                         [[0.1, False],
                          [-1, False],
                          [0, False],
                          [1, True],
                          [60, True],
                          ['foo', False]])
def test_tf_image_classification_config_file_max_trials(max_trials, valid):
    """
    Tests writing an Intel Neural Compressor config file with good and bad exit_policy_max_trials values
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, mock_dataset, batch_size=1, overwrite=True,
                                                exit_policy_max_trials=max_trials, tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, mock_dataset, batch_size=1, overwrite=True,
                                            exit_policy_max_trials=max_trials, tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.tensorflow
@pytest.mark.parametrize('seed,valid',
                         [[0.1, False],
                          [-1, False],
                          [0, True],
                          [1, True],
                          [123, True],
                          ['foo', False]])
def test_tf_image_classification_config_file_seed(seed, valid):
    """
    Tests writing an Intel Neural Compressor config file with good and bad tuning_random_seed values
    """
    try:
        temp_dir = tempfile.mkdtemp()
        nc_workspace = os.path.join(temp_dir, "nc_workspace")
        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            config_file = os.path.join(temp_dir, "config.yaml")
            mock_dataset.dataset_dir = "/tmp/data/my_photos"

            if not valid:
                with pytest.raises(ValueError):
                    model.write_inc_config_file(config_file, mock_dataset, batch_size=1, overwrite=True,
                                                tuning_random_seed=seed, tuning_workspace=nc_workspace)
            else:
                model.write_inc_config_file(config_file, mock_dataset, batch_size=1, overwrite=True,
                                            tuning_random_seed=seed, tuning_workspace=nc_workspace)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.tensorflow
def test_tf_image_classification_quantization():
    """
    Given valid directories for the saved model, output dir, and config file, test the quantization function with
    the actual Intel Neural Compressor called mocked out.
    """
    try:
        output_dir = tempfile.mkdtemp()
        saved_model_dir = tempfile.mkdtemp()
        saved_model_file = os.path.join(saved_model_dir, "saved_model.pb")
        Path(saved_model_file).touch()
        dummy_config_file = os.path.join(saved_model_dir, "config.yaml")
        Path(dummy_config_file).touch()

        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            with patch('neural_compressor.experimental.Quantization') as mock_q:
                mock_dataset.dataset_dir = "/tmp/data/my_photos"

                model.quantize(output_dir, dummy_config_file)
                mock_q.assert_called_with(dummy_config_file)
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)


@pytest.mark.tensorflow
def test_tf_image_classification_optimize_graph():
    """
    Given valid directories for the saved model, output dir, and config file, test the graph optimization function with
    the actual Intel Neural Compressorcalled mocked out.
    """
    try:
        output_dir = tempfile.mkdtemp()
        saved_model_dir = tempfile.mkdtemp()
        saved_model_file = os.path.join(saved_model_dir, "saved_model.pb")
        Path(saved_model_file).touch()

        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            with patch('neural_compressor.experimental.Graph_Optimization') as mock_o:
                mock_dataset.dataset_dir = "/tmp/data/my_photos"
                model.optimize_graph(output_dir)
                mock_o.assert_called()
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)


@pytest.mark.tensorflow
def test_tf_image_classification_benchmark_model_does_not_exist():
    """
    Verifies the error that gets raise if benchmarking is done with a model that does not exist
    """
    try:
        output_dir = tempfile.mkdtemp()
        dummy_config_file = os.path.join(output_dir, "config.yaml")
        Path(dummy_config_file).touch()
        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            mock_dataset.dataset_dir = "/tmp/data/my_photos"
            random_dir = str(uuid.uuid4())
            saved_model_dir = tempfile.mkdtemp()
            with patch('neural_compressor.experimental.Benchmark'):
                # It's not a directory, so we expect an error
                with pytest.raises(NotADirectoryError):
                    model.benchmark(random_dir, dummy_config_file)

                # An empty directory with no saved model should alos generate an error
                with pytest.raises(FileNotFoundError):
                    model.benchmark(saved_model_dir, dummy_config_file)

    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)


@pytest.mark.tensorflow
def test_tf_image_classification_inc_benchmark():
    """
    Verifies that if we have valid parameters for the saved model, config file, and mode, benchmarking is called. The
    actual benchmarking calls to Intel Neural Compressor are mocked out.
    """
    try:
        output_dir = tempfile.mkdtemp()
        saved_model_dir = tempfile.mkdtemp()
        saved_model_file = os.path.join(saved_model_dir, "saved_model.pb")
        Path(saved_model_file).touch()
        dummy_config_file = os.path.join(saved_model_dir, "config.yaml")
        Path(dummy_config_file).touch()

        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
            with patch('neural_compressor.experimental.Benchmark') as mock_bench:
                mock_dataset.dataset_dir = "/tmp/data/my_photos"

                model.benchmark(saved_model_dir, dummy_config_file)
                mock_bench.assert_called_with(dummy_config_file)
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)


@pytest.mark.tensorflow
@pytest.mark.parametrize('mode,valid',
                         [['abc', False],
                          [1, False],
                          [0, False],
                          ['performance', True],
                          ['accuracy', True]])
def test_tf_image_classification_inc_benchmark_mode(mode, valid):
    """
    Checks error handling for the benchmarking mode
    """
    try:
        output_dir = tempfile.mkdtemp()
        saved_model_dir = tempfile.mkdtemp()
        saved_model_file = os.path.join(saved_model_dir, "saved_model.pb")
        Path(saved_model_file).touch()
        dummy_config_file = os.path.join(saved_model_dir, "config.yaml")
        Path(dummy_config_file).touch()

        model = model_factory.get_model('efficientnet_b0', 'tensorflow')
        with patch('tlt.models.image_classification.tf_image_classification_model.TFCustomImageClassificationDataset') \
                as mock_dataset:
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
