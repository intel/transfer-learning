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
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:
    print("WARNING: Unable to import torch. Torch may not be installed")


import os
import pytest
import shutil
import tempfile
from unittest.mock import patch, MagicMock

from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.file_utils import download_and_extract_tar_file

try:
    from tlt.datasets.image_anomaly_detection.pytorch_custom_image_anomaly_detection_dataset import PyTorchCustomImageAnomalyDetectionDataset  # noqa: E501
    from tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model import extract_features
except ModuleNotFoundError:
    print("WARNING: Unable to import torch. Torch may not be installed")


@pytest.mark.integration
@pytest.mark.pytorch
class TestImageAnomalyDetectionCustomDataset:
    """
    Tests for PyTorch image anomaly detection using a custom dataset using the flowers dataset
    """
    @classmethod
    def setup_class(cls):
        os.makedirs('/tmp/data', exist_ok=True)
        temp_dir = tempfile.mkdtemp(dir='/tmp/data')
        custom_dataset_path = os.path.join(temp_dir, "flower_photos")

        if not os.path.exists(custom_dataset_path):
            download_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
            download_and_extract_tar_file(download_url, temp_dir)
            # Rename daisy to "good" and delete all but one other kind to make the dataset small
            os.rename(os.path.join(custom_dataset_path, 'daisy'), os.path.join(custom_dataset_path, 'good'))
            for flower in ['dandelion', 'roses', 'sunflowers']:
                shutil.rmtree(os.path.join(custom_dataset_path, flower))

        os.makedirs('/tmp/output', exist_ok=True)
        cls._output_dir = tempfile.mkdtemp(dir='/tmp/output')
        os.environ["TORCH_HOME"] = cls._output_dir
        cls._temp_dir = temp_dir
        cls._dataset_dir = custom_dataset_path

    @classmethod
    def teardown_class(cls):
        # remove directories
        for dir in [cls._output_dir, cls._temp_dir]:
            if os.path.exists(dir):
                print("Deleting test directory:", dir)
                shutil.rmtree(dir)

    @pytest.mark.parametrize('model_name',
                             ['resnet18'])
    def test_custom_dataset_workflow(self, model_name):
        """
        Tests the workflow for PYT image anomaly detection using a custom dataset
        """
        framework = 'pytorch'
        use_case = 'image_anomaly_detection'

        # Get the dataset
        dataset = dataset_factory.load_dataset(self._dataset_dir, use_case=use_case, framework=framework,
                                               shuffle_files=False)
        assert ['tulips'] == dataset.defect_names
        assert ['bad', 'good'] == dataset.class_names

        # Get the model
        model = model_factory.get_model(model_name, framework, use_case)

        # Preprocess the dataset and split to get small subsets for training and validation
        dataset.preprocess(model.image_size, 32)
        dataset.shuffle_split(train_pct=0.5, val_pct=0.5, seed=10)

        # Train for 1 epoch
        pca_components, trained_model = model.train(dataset, self._output_dir,
                                                    layer_name='layer3', seed=10, simsiam=False)

        # Extract features
        images, labels = dataset.get_batch(subset='validation')
        features = extract_features(trained_model, images, layer_name='layer3', pooling=['avg', 2])
        assert len(features) == 32

        # Evaluate
        threshold, auroc = model.evaluate(dataset)
        assert isinstance(auroc, float)

        # Predict with a batch
        predictions = model.predict(images)
        assert len(predictions) == 32

        # Export the saved model
        saved_model_dir = model.export(self._output_dir)
        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "model.pt"))

        # Reload the saved model
        reload_model = model_factory.get_model(model_name, framework, use_case)
        reload_model.load_from_directory(saved_model_dir)

        # Evaluate reloaded model
        reload_threshold, reload_auroc = reload_model.evaluate(dataset)
        assert reload_threshold == threshold
        assert reload_auroc == auroc

    @pytest.mark.parametrize('model_name,device',
                             [['resnet18', 'hpu']])
    def test_no_hpu_workflow(self, model_name, device):
        """
        Tests the workflow for PYT image anomaly detection when "hpu" is specified but not available
        """
        framework = 'pytorch'
        use_case = 'image_anomaly_detection'

        # Get the dataset
        dataset = dataset_factory.load_dataset(self._dataset_dir, use_case=use_case, framework=framework,
                                               shuffle_files=False)
        assert ['tulips'] == dataset.defect_names
        assert ['bad', 'good'] == dataset.class_names

        # Get the model
        model = model_factory.get_model(model_name, framework, use_case, device=device)

        # Preprocess the dataset and split to get small subsets for training and validation
        dataset.preprocess(model.image_size, 32)
        dataset.shuffle_split(train_pct=0.1, val_pct=0.1, seed=10)

        # Train for 1 epoch
        pca_components, trained_model = model.train(dataset, self._output_dir,
                                                    layer_name='layer3', seed=10, simsiam=False)
        assert model._device == "cpu"

        # Extract features
        images, labels = dataset.get_batch(subset='validation')
        features = extract_features(trained_model, images, layer_name='layer3', pooling=['avg', 2])
        assert len(features) == 32

        # Evaluate
        threshold, auroc = model.evaluate(dataset)
        assert model._device == "cpu"
        assert isinstance(auroc, float)

        # Predict with a batch
        predictions = model.predict(images)
        assert model._device == "cpu"
        assert len(predictions) == 32

        # Export the saved model
        saved_model_dir = model.export(self._output_dir)
        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "model.pt"))

        # Reload the saved model
        reload_model = model_factory.get_model(model_name, framework, use_case, device=device)
        reload_model.load_from_directory(saved_model_dir)

        # Evaluate reloaded model
        reload_threshold, reload_auroc = reload_model.evaluate(dataset)
        assert model._device == "cpu"
        assert reload_threshold == threshold
        assert reload_auroc == auroc

    @pytest.mark.parametrize('model_name,device',
                             [['resnet18', 'hpu']])
    @patch('tlt.models.image_anomaly_detection.utils.htcore', create=True, new_callable=lambda: MagicMock(name='htcore'))  # noqa: E501
    @patch("tlt.models.image_anomaly_detection.utils.torch")
    @patch("tlt.models.image_anomaly_detection.utils.ProgressMeter")
    @patch("tlt.models.image_anomaly_detection.utils.is_hpu_available")
    @patch("tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model.is_hpu_available")
    @patch("tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model.PyTorchImageAnomalyDetectionModel.load_checkpoint_weights")  # noqa: E501
    @patch("tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model.torch")
    @patch("tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model.nn")
    @patch("tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model.builder")
    @patch("tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model.pca")
    @patch("tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model.get_feature_extraction_model")
    def test_hpu_workflow(self, mock_extraction, mock_pca, mock_builder, mock_nn, mock_torch, mock_load_ckpt,
                          mock_model_hpu_available, mock_hpu_available, mock_prog, mock_utils_torch, mock_habana,
                          model_name, device):
        """
        Tests the workflow for PYT image anomaly detection when "hpu" is the selected device and is available
        """
        framework = 'pytorch'
        use_case = 'image_anomaly_detection'

        # Mock out the model downloader since we aren't doing a real train, we don't need a real model
        mock_model = MagicMock()
        mock_model.__call__ = MagicMock(return_value=(1, 2, 3, 4))
        mock_model.to = MagicMock()
        mock_model.to.return_value = MagicMock(return_value=(1, 2, 3, 4))
        mock_load_ckpt.return_value = mock_model
        mock_builder.SimSiam = MagicMock()
        mock_builder.SimSiam.return_value = mock_model

        # Get the model
        model = model_factory.get_model(model_name, framework, use_case, device=device)
        model.load_pretrained_model = MagicMock()

        # Create a mock dataset, since we aren't really training
        mock_dataset = MagicMock()
        mock_dataset.__class__ = PyTorchCustomImageAnomalyDetectionDataset
        mock_dataset.get_batch.return_value = MagicMock(content=[1, 2]), MagicMock(content=[1, 2])
        mock_dataset._train_loader = MagicMock()
        mock_inputs = MagicMock()
        data_loader_data = [(mock_inputs, 0), (mock_inputs, 0)]
        mock_dataset._train_loader.__iter__ = MagicMock(return_value=iter(data_loader_data))

        # Train for 1 epoch
        model.train(mock_dataset, self._output_dir, layer_name='layer3', seed=10, simsiam=True)

        if device == 'hpu':
            # mark_step is called before and after each step
            assert mock_habana.mark_step.call_count == len(data_loader_data) * 2
        else:
            mock_habana.mark_step.assert_not_called()

    def test_custom_model_workflow(self):
        """
        Tests the workflow for PYT image anomaly detection using a custom model and custom dataset
        """
        framework = 'pytorch'
        use_case = 'image_anomaly_detection'

        # Get the dataset
        dataset = dataset_factory.load_dataset(self._dataset_dir, use_case=use_case, framework=framework,
                                               shuffle_files=False)

        # Define a model
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(functional.relu(self.conv1(x)))
                x = self.pool(functional.relu(self.conv2(x)))
                x = torch.flatten(x, 1)
                x = functional.relu(self.fc1(x))
                x = functional.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        net = Net()

        # Load the model
        model = model_factory.load_model('custom_model', net, framework=framework, use_case=use_case)
        model.list_layers()

        # Preprocess the dataset and split to get small subsets for training and validation
        dataset.preprocess(image_size=224, batch_size=32)
        dataset.shuffle_split(train_pct=0.5, val_pct=0.5, seed=10)

        # Train for 1 epoch
        pca_components, trained_model = model.train(dataset, self._output_dir,
                                                    layer_name='conv2', seed=10, simsiam=False)

        # Extract features
        images, labels = dataset.get_batch(subset='validation')
        features = extract_features(trained_model, images, layer_name='conv2', pooling=['avg', 2])
        assert len(features) == 32

        # Evaluate
        threshold, auroc = model.evaluate(dataset)
        assert isinstance(auroc, float)

        # Predict with a batch
        predictions = model.predict(images)
        assert len(predictions) == 32

    @pytest.mark.parametrize('model_name',
                             ['resnet18'])
    def test_simsiam_workflow_quantization(self, model_name):
        """
        Tests the workflow for PYT image anomaly detection using a custom dataset
        and simsiam feature extractor enabled
        """
        framework = 'pytorch'
        use_case = 'image_anomaly_detection'

        # Get the dataset
        dataset = dataset_factory.load_dataset(self._dataset_dir, use_case=use_case, framework=framework,
                                               shuffle_files=False)
        assert ['tulips'] == dataset.defect_names
        assert ['bad', 'good'] == dataset.class_names

        # Get the model
        model = model_factory.get_model(model_name, framework, use_case)

        # Preprocess the dataset and split to get small subsets for training and validation
        dataset.preprocess(model.image_size, 32)
        dataset.shuffle_split(train_pct=0.5, val_pct=0.5, seed=10)

        # Train for 1 epoch
        pca_components, trained_model = model.train(dataset, self._output_dir, epochs=1,
                                                    layer_name='layer3', feature_dim=1000, pred_dim=250,
                                                    seed=10, simsiam=True, initial_checkpoints=None)

        # Evaluate
        threshold, auroc = model.evaluate(dataset)
        assert isinstance(auroc, float)

        # Predict with a batch
        images, labels = dataset.get_batch(subset='validation')
        predictions = model.predict(images)
        assert len(predictions) == 32

        # Quantization
        inc_output_dir = os.path.join(self._output_dir, "quantized", model_name)
        os.makedirs(inc_output_dir, exist_ok=True)
        model.quantize(inc_output_dir, dataset)
        assert os.path.exists(os.path.join(inc_output_dir, "model.pt"))

    @pytest.mark.parametrize('model_name',
                             ['resnet18'])
    def test_cutpaste_workflow_benchmark(self, model_name):
        """
        Tests the workflow for PYT image anomaly detection using a custom dataset
        and cutpaste feature extractor enabled
        """
        framework = 'pytorch'
        use_case = 'image_anomaly_detection'

        # Get the dataset
        dataset = dataset_factory.load_dataset(self._dataset_dir, use_case=use_case, framework=framework,
                                               shuffle_files=False)
        assert ['tulips'] == dataset.defect_names
        assert ['bad', 'good'] == dataset.class_names

        # Get the model
        model = model_factory.get_model(model_name, framework, use_case)

        # Preprocess the dataset and split to get small subsets for training and validation
        dataset.preprocess(model.image_size, 32)
        dataset.shuffle_split(train_pct=0.5, val_pct=0.25, test_pct=0.25, seed=10)

        # Train for 1 epoch
        pca_components, trained_model = model.train(dataset, self._output_dir, epochs=1,
                                                    layer_name='layer3', optim='sgd', freeze_resnet=20,
                                                    head_layer=2, cutpaste_type='normal', seed=10,
                                                    cutpaste=True)

        # Evaluate
        threshold, auroc = model.evaluate(dataset, use_test_set=True)
        assert isinstance(auroc, float)

        # Predict with a batch
        images, labels = dataset.get_batch(subset='test')
        predictions = model.predict(images, pca_mats=pca_components)
        assert len(predictions) == 32

        # Export the saved model
        saved_model_dir = model.export(self._output_dir)
        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "model.pt"))

        # Benchmark
        model.benchmark(dataset=dataset)

    @pytest.mark.parametrize('model_name',
                             ['vgg11',
                              'efficientnet_b1',
                              'alexnet'])
    def test_no_simsiam_or_cutpaste(self, model_name):
        """
        Tests the workflow for PYT image anomaly detection using a custom dataset
        and cutpaste feature extractor enabled
        """
        framework = 'pytorch'
        use_case = 'image_anomaly_detection'

        # Get the dataset
        dataset = dataset_factory.load_dataset(self._dataset_dir, use_case=use_case, framework=framework,
                                               shuffle_files=False)
        assert ['tulips'] == dataset.defect_names
        assert ['bad', 'good'] == dataset.class_names

        # Get the model
        model = model_factory.get_model(model_name, framework, use_case)

        # Preprocess the dataset and split to get small subsets for training and validation
        dataset.preprocess(model.image_size, 32)
        dataset.shuffle_split(train_pct=0.5, val_pct=0.25, test_pct=0.25, seed=10)

        # Train for 1 epoch
        pca_components, trained_model = model.train(dataset, self._output_dir, epochs=1, kernel_size=2,
                                                    layer_name='features', pooling='avg', pca_threshold=0.99,
                                                    seed=10)

        # Evaluate
        threshold, auroc = model.evaluate(dataset, use_test_set=True)
        assert isinstance(auroc, float)

        # Predict with a batch
        images, labels = dataset.get_batch(subset='test')
        predictions = model.predict(images, pca_mats=pca_components)
        assert len(predictions) == 32

        # Export the saved model
        saved_model_dir = model.export(self._output_dir)
        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "model.pt"))
