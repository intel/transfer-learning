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
import numpy as np
import pytest
import shutil
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as functional

from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.file_utils import download_and_extract_tar_file


@pytest.mark.integration
@pytest.mark.pytorch
@pytest.mark.parametrize('model_name,dataset_name,extra_layers,correct_num_layers',
                         [['efficientnet_b0', 'CIFAR10', None, 2],
                          ['resnet18_ssl', 'CIFAR10', None, 1],
                          ['efficientnet_b0', 'CIFAR10', [1024, 512], 6],
                          ['resnet18', 'CIFAR10', [1024, 512], 5]])
def test_pyt_image_classification(model_name, dataset_name, extra_layers, correct_num_layers):
    """
    Tests basic transfer learning functionality for PyTorch image classification models using a torchvision dataset
    """
    framework = 'pytorch'
    output_dir = tempfile.mkdtemp()
    os.environ["TORCH_HOME"] = output_dir

    # Get the dataset
    dataset = dataset_factory.get_dataset('/tmp/data', 'image_classification', framework, dataset_name,
                                          'torchvision', split=["train"], shuffle_files=False)

    # Get the model
    model = model_factory.get_model(model_name, framework)

    # Preprocess the dataset
    dataset.preprocess(image_size='variable', batch_size=32)
    dataset.shuffle_split(train_pct=0.05, val_pct=0.05, seed=10)
    assert dataset._validation_type == 'shuffle_split'

    # Evaluate before training
    pretrained_metrics = model.evaluate(dataset)
    assert len(pretrained_metrics) > 0

    # Train
    model.train(dataset, output_dir=output_dir, epochs=1, do_eval=False, extra_layers=extra_layers, seed=10)
    assert len(list(model._model.children())[-1]) == correct_num_layers

    # Evaluate
    trained_metrics = model.evaluate(dataset)
    assert trained_metrics[0] <= pretrained_metrics[0]  # loss
    assert trained_metrics[1] >= pretrained_metrics[1]  # accuracy

    # Predict with a batch
    images, labels = dataset.get_batch()
    predictions = model.predict(images)
    assert len(predictions) == 32
    probabilities = model.predict(images, return_type='probabilities')
    assert probabilities.shape == torch.Size([32, 10])  # CIFAR has 10 classes
    np.testing.assert_almost_equal(torch.sum(probabilities), np.float32(32), decimal=4)

    # Export the saved model
    saved_model_dir = model.export(output_dir)
    assert os.path.isdir(saved_model_dir)
    assert os.path.isfile(os.path.join(saved_model_dir, "model.pt"))

    # Reload the saved model
    reload_model = model_factory.get_model(model_name, framework)
    reload_model.load_from_directory(saved_model_dir)

    # Evaluate
    reload_metrics = reload_model.evaluate(dataset)
    assert reload_metrics == trained_metrics

    # Ensure we get not implemented errors for graph_optimization
    with pytest.raises(NotImplementedError):
        model.optimize_graph(saved_model_dir, os.path.join(saved_model_dir, 'optimized'))

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    # Ensure we get not implemented errors for quantization
    inc_config_file_path = os.path.join(output_dir, "pytorch_{}.yaml".format(model_name))
    with pytest.raises(NotImplementedError):
        model.write_inc_config_file(inc_config_file_path, dataset, batch_size=32)


@pytest.mark.pytorch
def test_pyt_image_classification_custom_model():
    """
    Tests basic transfer learning functionality for custom PyTorch image classification models using a Torchvision
    dataset
    """
    framework = 'pytorch'
    use_case = 'image_classification'
    output_dir = tempfile.mkdtemp()
    os.environ["TORCH_HOME"] = output_dir

    # Get the dataset
    dataset = dataset_factory.get_dataset('/tmp/data', 'image_classification', framework, 'CIFAR10',
                                          'torchvision', split=["train"], shuffle_files=False)

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

    # Get the model
    model = model_factory.load_model('custom_model', net, framework, use_case)
    assert model.num_classes == 10

    # Preprocess the dataset
    dataset.preprocess(image_size='variable', batch_size=32)
    dataset.shuffle_split(train_pct=0.05, val_pct=0.05, seed=10)
    assert dataset._validation_type == 'shuffle_split'

    # Train
    model.train(dataset, output_dir=output_dir, epochs=1, do_eval=False, seed=10)

    # Evaluate
    trained_metrics = model.evaluate(dataset)
    assert trained_metrics[0] > 0.0  # loss
    assert trained_metrics[1] > 0.0  # accuracy

    # Predict with a batch
    images, labels = dataset.get_batch()
    predictions = model.predict(images)
    assert len(predictions) == 32
    probabilities = model.predict(images, return_type='probabilities')
    assert probabilities.shape == torch.Size([32, 10])  # CIFAR has 10 classes
    np.testing.assert_almost_equal(torch.sum(probabilities), np.float32(32), decimal=4)

    # Export the saved model
    saved_model_dir = model.export(output_dir)
    assert os.path.isdir(saved_model_dir)
    assert os.path.isfile(os.path.join(saved_model_dir, "model.pt"))

    # Reload the saved model
    reload_model = model_factory.load_model('custom_model', saved_model_dir, framework, use_case)

    # Evaluate
    reload_metrics = reload_model.evaluate(dataset)
    assert reload_metrics == trained_metrics

    # Ensure we get not implemented errors for graph_optimization
    with pytest.raises(NotImplementedError):
        model.optimize_graph(saved_model_dir, os.path.join(saved_model_dir, 'optimized'))

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    # Ensure we get not implemented errors for quantization
    inc_config_file_path = os.path.join(output_dir, "pytorch_{}.yaml".format('custom_model'))
    with pytest.raises(NotImplementedError):
        model.write_inc_config_file(inc_config_file_path, dataset, batch_size=32)


class TestImageClassificationCustomDataset:
    """
    Tests for PyTorch image classification using a custom dataset using the flowers dataset
    """
    @classmethod
    def setup_class(cls):
        temp_dir = tempfile.mkdtemp(dir='/tmp/data')
        custom_dataset_path = os.path.join(temp_dir, "flower_photos")

        if not os.path.exists(custom_dataset_path):
            download_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
            download_and_extract_tar_file(download_url, temp_dir)

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

    @pytest.mark.pytorch
    @pytest.mark.parametrize('model_name,add_aug,ipex_optimize',
                             [['efficientnet_b0', ['hflip'], True],
                              ['resnet18', ['rotate'], True],
                              ['resnet18_ssl', ['rotate'], True],
                              ['vit_b_16', None, False]])
    def test_custom_dataset_workflow(self, model_name, add_aug, ipex_optimize):
        """
        Tests the full workflow for PYT image classification using a custom dataset
        """
        framework = 'pytorch'
        use_case = 'image_classification'

        # Get the dataset
        dataset = dataset_factory.load_dataset(self._dataset_dir, use_case=use_case, framework=framework,
                                               shuffle_files=False)
        assert ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'] == dataset.class_names

        # Get the model
        model = model_factory.get_model(model_name, framework)

        # Preprocess the dataset and split to get small subsets for training and validation
        dataset.preprocess(model.image_size, 32, add_aug=add_aug)
        dataset.shuffle_split(train_pct=0.1, val_pct=0.1, seed=10)

        # Train for 1 epoch
        model.train(dataset, output_dir=self._output_dir, epochs=1, do_eval=False, seed=10, ipex_optimize=ipex_optimize)

        # Evaluate
        model.evaluate(dataset)

        # Predict with a batch
        images, labels = dataset.get_batch()
        predictions = model.predict(images)
        assert len(predictions) == 32

        # export the saved model
        saved_model_dir = model.export(self._output_dir)
        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "model.pt"))

        # Reload the saved model
        reload_model = model_factory.get_model(model_name, framework)
        reload_model.load_from_directory(saved_model_dir)

        # Evaluate
        metrics = reload_model.evaluate(dataset)
        assert len(metrics) > 0

        # Test benchmarking and quantization with non-IPEX ResNet18
        if model_name == "resnet18" and not ipex_optimize:
            inc_config_file_path = os.path.join(self._output_dir, "pyt_{}.yaml".format(model_name))
            nc_workspace = os.path.join(self._output_dir, "nc_workspace")
            model.write_inc_config_file(inc_config_file_path, dataset, batch_size=32, overwrite=True,
                                        accuracy_criterion_relative=0.1, exit_policy_max_trials=10,
                                        exit_policy_timeout=0, tuning_workspace=nc_workspace)
            model.benchmark(saved_model_dir, inc_config_file_path, model_type='fp32')
            quantization_output = os.path.join(self._output_dir, "quantized", model_name)
            os.makedirs(quantization_output, exist_ok=True)
            model.quantize(saved_model_dir, quantization_output, inc_config_file_path)
            assert os.path.exists(os.path.join(quantization_output, "model.pt"))
            model.benchmark(quantization_output, inc_config_file_path, model_type='int8')


@pytest.mark.integration
@pytest.mark.pytorch
@pytest.mark.parametrize('model_name,dataset_name,epochs,lr,do_eval,early_stopping,lr_decay,final_lr,final_acc',
                         [['efficientnet_b0', 'CIFAR10', 10, 0.005, True, False, True, 0.001, 0.9888],
                          ['resnet18', 'CIFAR10', 1, 0.005, True, False, False, None, 0.2688],
                          ['efficientnet_b0', 'CIFAR10', 1, 0.001, False, False, False, None, 0.1976],
                          ['efficientnet_b0', 'CIFAR10', 10, 0.001, True, True, True, 0.0002, 0.8768]])
def test_pyt_image_classification_with_lr_options(model_name, dataset_name, epochs, lr, do_eval, early_stopping,
                                                  lr_decay, final_lr, final_acc):
    """
    Tests transfer learning for PyTorch image classification models using learning rate options
    """
    framework = 'pytorch'
    output_dir = tempfile.mkdtemp()
    os.environ["TORCH_HOME"] = output_dir

    # Get the dataset
    dataset = dataset_factory.get_dataset('/tmp/data', 'image_classification', framework, dataset_name,
                                          'torchvision', split=["train"], shuffle_files=False)

    # Get the model
    model = model_factory.get_model(model_name, framework)
    model.learning_rate = lr

    # Preprocess the dataset
    dataset.shuffle_split(train_pct=0.05, val_pct=0.05, shuffle_files=False)
    dataset.preprocess(image_size='variable', batch_size=32)
    assert dataset._validation_type == 'shuffle_split'

    # Train
    history = model.train(dataset, output_dir=output_dir, epochs=epochs, do_eval=do_eval, early_stopping=early_stopping,
                          lr_decay=lr_decay, seed=10)

    if final_lr:
        assert model._lr_scheduler.optimizer.param_groups[0]['lr'] == final_lr
    else:
        assert model._lr_scheduler is None

    assert history['Acc'][-1] == final_acc


@pytest.mark.pytorch
def test_pyt_freeze():
    """
    Tests layer freezing functionality for PyTorch image classification models using a torchvision dataset
    """
    dataset_name = 'CIFAR10'
    framework = 'pytorch'
    layer_name = 'features'
    model_name = 'efficientnet_b0'
    output_dir = tempfile.mkdtemp()
    os.environ["TORCH_HOME"] = output_dir

    # Get the dataset
    dataset = dataset_factory.get_dataset('/tmp/data', 'image_classification', framework, dataset_name,
                                          'torchvision', split=["train"], shuffle_files=False)
    # Get the model
    model = model_factory.get_model(model_name, framework)

    # Preprocess the dataset
    dataset.preprocess(image_size='variable', batch_size=32)
    dataset.shuffle_split(train_pct=0.05, val_pct=0.05, seed=10)

    # Train
    model.train(dataset, output_dir=output_dir, epochs=1, do_eval=False)

    # Check that everything in the layer is unfrozen
    model.unfreeze_layer("features")

    # Unfreeze everything in the layer
    for (name, module) in model._model.named_children():
        if name == layer_name:
            for layer in module.children():
                for param in layer.parameters():
                    assert param.requires_grad is True

    # Check that everything in the layer is frozen
    model.freeze_layer("features")

    # Freeze everything in the layer
    for (name, module) in model._model.named_children():
        if name == layer_name:
            for layer in module.children():
                for param in layer.parameters():
                    assert param.requires_grad is False

    # Test functionality of list_layers()
    trainable_params = model.list_layers()
    assert trainable_params == 12810  # Number of trainable params in efficientnet_b0
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
