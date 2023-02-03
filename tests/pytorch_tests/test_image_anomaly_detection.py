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


class TestImageAnomalyDetectionCustomDataset:
    """
    Tests for PyTorch image anomaly detection using a custom dataset using the flowers dataset
    """
    @classmethod
    def setup_class(cls):
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

    @pytest.mark.integration
    @pytest.mark.pytorch
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

        # Extract features
        images, labels = dataset.get_batch(subset='validation')
        features = model.extract_features(images, 'layer3', ['avg', 2])
        assert len(features) == 32

        # Train for 1 epoch
        model.train(dataset, output_dir=self._output_dir, do_eval=False, seed=10)

        # Evaluate
        auroc = model.evaluate(dataset)
        assert isinstance(auroc, float)

        # Predict with a batch
        predictions = model.predict(images)
        assert len(predictions) == 32
