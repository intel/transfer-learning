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
# SPDX-License-Identifier: EPL-2.0
#

import os
import pytest
import shutil
import tempfile

from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.file_utils import download_and_extract_tar_file


@pytest.mark.pytorch
@pytest.mark.parametrize('model_name,dataset_name',
                         [['efficientnet_b0', 'CIFAR10'],
                          ['resnet18', 'CIFAR10']])
def test_pyt_image_classification(model_name, dataset_name):
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
    model.train(dataset, output_dir=output_dir, epochs=1, do_eval=False)

    # Evaluate
    trained_metrics = model.evaluate(dataset)
    assert trained_metrics[0] <= pretrained_metrics[0]  # loss
    assert trained_metrics[1] >= pretrained_metrics[1]  # accuracy

    # Predict with a batch
    images, labels = dataset.get_batch()
    predictions = model.predict(images)
    assert len(predictions) == 32

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
    with pytest.raises(NotImplementedError) as e:
        model.optimize_graph(saved_model_dir, os.path.join(saved_model_dir, 'optimized'))

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    # Ensure we get not implemented errors for quantization
    inc_config_file_path = os.path.join(output_dir, "pytorch_{}.yaml".format(model_name))
    with pytest.raises(NotImplementedError) as e:
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
    @pytest.mark.parametrize('model_name',
                             ['efficientnet_b0',
                              'resnet18'])
    def test_custom_dataset_workflow(self, model_name):
        """
        Tests the full workflow for PYT image classification using a custom dataset
        """
        framework = 'pytorch'
        use_case = 'image_classification'

        # Get the dataset
        dataset = dataset_factory.load_dataset(self._dataset_dir, use_case=use_case, framework=framework, shuffle_files=False)
        assert ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'] == dataset.class_names

        # Get the model
        model = model_factory.get_model(model_name, framework)

        # Preprocess the dataset and split to get small subsets for training and validation
        dataset.preprocess(model.image_size, 32)
        dataset.shuffle_split(train_pct=0.1, val_pct=0.1, seed=10)

        # Train for 1 epoch
        model.train(dataset, output_dir=self._output_dir, epochs=1, do_eval=False)

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

