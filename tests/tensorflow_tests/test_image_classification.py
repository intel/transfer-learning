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
import numpy as np
from tensorflow import keras

from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.file_utils import download_and_extract_tar_file


@pytest.mark.tensorflow
@pytest.mark.parametrize('model_name,dataset_name,train_accuracy,retrain_accuracy,extra_layers,correct_num_layers',
                         [['efficientnet_b0', 'tf_flowers', 0.34375, 0.5625, None, 2],
                          ['resnet_v1_50', 'tf_flowers', 0.40625, 0.6875, None, 2],
                          ['efficientnet_b0', 'tf_flowers', 0.8125, 0.96875, [1024, 512], 4]])
def test_tf_image_classification(model_name, dataset_name, train_accuracy, retrain_accuracy, extra_layers,
                                 correct_num_layers):
    """
    Tests basic transfer learning functionality for TensorFlow image classification models using TF Datasets
    """
    framework = 'tensorflow'
    use_case = 'image_classification'
    output_dir = tempfile.mkdtemp()

    # Get the dataset
    dataset = dataset_factory.get_dataset('/tmp/data', 'image_classification', framework, dataset_name,
                                          'tf_datasets', split=["train[:5%]"])

    # Get the model
    model = model_factory.get_model(model_name, framework)

    # Preprocess the dataset
    dataset.preprocess(model.image_size, 32)
    dataset.shuffle_split(seed=10)

    # Evaluate before training
    pretrained_metrics = model.evaluate(dataset)
    assert len(pretrained_metrics) > 0

    # Train
    history = model.train(dataset, output_dir=output_dir, epochs=1, shuffle_files=False, seed=10, do_eval=False,
                          extra_layers=extra_layers)
    assert history is not None
    assert history['acc'] == [train_accuracy]
    assert len(model._model.layers) == correct_num_layers

    # Verify that checkpoints were generated
    checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(model_name))
    assert os.path.isdir(checkpoint_dir)
    assert len(os.listdir(checkpoint_dir))

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
    assert os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb"))

    # Reload the saved model
    reload_model = model_factory.get_model(model_name, framework)
    reload_model.load_from_directory(saved_model_dir)

    # Evaluate
    reload_metrics = reload_model.evaluate(dataset)
    assert reload_metrics == trained_metrics

    # Optimize the graph
    if model_name == 'resnet_v1_50':
        optimized_model_dir = os.path.join(output_dir, "optimized")
        model.optimize_graph(saved_model_dir, optimized_model_dir)
        assert os.path.isfile(os.path.join(optimized_model_dir, "saved_model.pb"))

    # Test generating an INC config file (not implemented yet for TFDS)
    inc_config_file_path = os.path.join(output_dir, "tf_{}.yaml".format(model_name))
    with pytest.raises(NotImplementedError) as e:
        model.write_inc_config_file(inc_config_file_path, dataset, batch_size=32, tuning_workspace=output_dir)

    # Retrain from checkpoints and verify that we have better accuracy than the original training
    retrain_model = model_factory.load_model(model_name, saved_model_dir, framework, use_case)
    retrain_history = retrain_model.train(dataset, output_dir=output_dir, epochs=1, initial_checkpoints=checkpoint_dir,
                                          shuffle_files=False, seed=10, do_eval=False)
    assert retrain_history['acc'] == [retrain_accuracy]

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

@pytest.mark.tensorflow
def test_tf_image_classification_custom_model():
    """
    Tests basic transfer learning functionality for a custom TensorFlow image classification model using TF Datasets
    """
    framework = 'tensorflow'
    use_case = 'image_classification'
    output_dir = tempfile.mkdtemp()
    model_name = 'custom_model'
    image_size = 227

    # Get the dataset
    dataset = dataset_factory.get_dataset('/tmp/data', 'image_classification', framework, 'tf_flowers',
                                          'tf_datasets', split=["train[:5%]"])

    # Define a custom model
    alexnet = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu',
                            input_shape=(image_size,image_size,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation='softmax')
    ])

    model = model_factory.load_model(model_name=model_name, model=alexnet, framework=framework, use_case=use_case)
    assert model.num_classes == 5
    assert model._image_size == 227

    # Preprocess the dataset
    dataset.preprocess(image_size, 32)
    dataset.shuffle_split(seed=10)

    # Train
    history = model.train(dataset, output_dir=output_dir, epochs=1, shuffle_files=False, seed=10)
    assert history is not None

    # Verify that checkpoints were generated
    checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(model_name))
    assert os.path.isdir(checkpoint_dir)
    assert len(os.listdir(checkpoint_dir))

    # Evaluate
    trained_metrics = model.evaluate(dataset)
    assert trained_metrics is not None

    # Predict with a batch
    images, labels = dataset.get_batch()
    predictions = model.predict(images)
    assert len(predictions) == 32

    # Export the saved model
    saved_model_dir = model.export(output_dir)
    assert os.path.isdir(saved_model_dir)
    assert os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb"))

    # Reload the saved model
    reload_model = model_factory.load_model(model_name, saved_model_dir, framework, use_case)

    # Evaluate
    reload_metrics = reload_model.evaluate(dataset)
    assert reload_metrics == trained_metrics

    # Retrain from checkpoints and verify that we have better accuracy than the original training
    retrain_model = model_factory.load_model(model_name, saved_model_dir, framework, use_case)
    retrain_history = retrain_model.train(dataset, output_dir=output_dir, epochs=1, initial_checkpoints=checkpoint_dir,
                                          shuffle_files=False, seed=10)
    assert retrain_history is not None

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)


class TestImageClassificationCustomDataset:
    """
    Tests for TensorFlow image classification using a custom dataset using the flowers dataset
    """
    @classmethod
    def setup_class(cls):
        temp_dir = tempfile.mkdtemp(dir='/tmp/data')
        custom_dataset_path = os.path.join(temp_dir, "flower_photos")

        if not os.path.exists(custom_dataset_path):
            download_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
            download_and_extract_tar_file(download_url, temp_dir)

        cls._output_dir = tempfile.mkdtemp()
        cls._temp_dir = temp_dir
        cls._dataset_dir = custom_dataset_path

    @classmethod
    def teardown_class(cls):
        # remove directories
        for dir in [cls._output_dir, cls._temp_dir]:
            if os.path.exists(dir):
                print("Deleting test directory:", dir)
                shutil.rmtree(dir)

    @pytest.mark.tensorflow
    @pytest.mark.parametrize('model_name,train_accuracy,retrain_accuracy',
                             [['efficientnet_b0', 0.9375, 0.9375],
                              ['resnet_v1_50', 1.0, 1.0]])
    def test_custom_dataset_workflow(self, model_name, train_accuracy, retrain_accuracy):
        """
        Tests the full workflow for TF image classification using a custom dataset
        """
        framework = 'tensorflow'
        use_case = 'image_classification'

        # Get the dataset
        dataset = dataset_factory.load_dataset(self._dataset_dir, use_case=use_case, framework=framework,
                                               shuffle_files=False)
        assert ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'] == dataset.class_names

        # Get the model
        model = model_factory.get_model(model_name, framework)

        # Preprocess the dataset and split to get small subsets for training and validation
        dataset.preprocess(model.image_size, 32)
        dataset.shuffle_split(train_pct=0.1, val_pct=0.1, seed=10)

        # Train for 1 epoch
        history = model.train(dataset, output_dir=self._output_dir, epochs=1, shuffle_files=False, seed=10,
                              do_eval=False)
        assert history is not None
        assert history['acc'] == [train_accuracy]

        # Verify that checkpoints were generated
        checkpoint_dir = os.path.join(self._output_dir, "{}_checkpoints".format(model_name))
        assert os.path.isdir(checkpoint_dir)
        assert len(os.listdir(checkpoint_dir))

        # Evaluate
        model.evaluate(dataset)

        # Predict with a batch
        images, labels = dataset.get_batch()
        predictions = model.predict(images)
        assert len(predictions) == 32

        # export the saved model
        saved_model_dir = model.export(self._output_dir)
        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb"))

        # Reload the saved model
        reload_model = model_factory.get_model(model_name, framework)
        reload_model.load_from_directory(saved_model_dir)

        # Evaluate
        metrics = reload_model.evaluate(dataset)
        assert len(metrics) > 0

        # Retrain from checkpoints and verify that we have better accuracy than the original training
        retrain_model = model_factory.get_model(model_name, framework)
        retrain_history = retrain_model.train(dataset, output_dir=self._output_dir, epochs=1,
                                              initial_checkpoints=checkpoint_dir, shuffle_files=False, seed=10,
                                              do_eval=False)
        assert retrain_history['acc'] == [retrain_accuracy]

        # Test benchmarking, quantization, and graph optimization with ResNet50
        if model_name == "resnet_v1_50":
            inc_config_file_path = os.path.join(self._output_dir, "tf_{}.yaml".format(model_name))
            nc_workspace = os.path.join(self._output_dir, "nc_workspace")
            model.write_inc_config_file(inc_config_file_path, dataset, batch_size=32, accuracy_criterion_relative=0.1,
                                        exit_policy_max_trials=10, exit_policy_timeout=0, tuning_workspace=nc_workspace)

            quantization_output = os.path.join(self._output_dir, "quantized", model_name)
            os.makedirs(quantization_output)
            model.quantize(saved_model_dir, quantization_output, inc_config_file_path)
            assert os.path.exists(os.path.join(quantization_output, "saved_model.pb"))
            model.benchmark(quantization_output, inc_config_file_path)

@pytest.mark.tensorflow
@pytest.mark.parametrize('model_name,dataset_name,epochs,learning_rate,do_eval,early_stopping,lr_decay,accuracy,\
                          val_accuracy,lr_final',
                         [['efficientnet_b0', 'tf_flowers', 4, 0.001, False, False, False, 0.875, None, 0.001],
                          ['efficientnet_b0', 'tf_flowers', 4, 0.001, True, False, False, 0.875, 0.65625, 0.001],
                          ['efficientnet_b0', 'tf_flowers', 4, 0.001, True, False, True, 0.875, 0.65625, 0.001],
                          ['efficientnet_b0', 'tf_flowers', 4, 0.001, False, False, True, 0.875, None, 0.001],
                          ['efficientnet_b0', 'tf_flowers', 16, 0.005, True, False, True, 1.0, 0.8125, 1.0000e-03],
                          ['efficientnet_b0', 'tf_flowers', 25, 0.001, True, True, False, 1.0, 0.8125, 0.0002]])
def test_tf_image_classification_with_lr_options(model_name, dataset_name, epochs, learning_rate, do_eval, early_stopping,
                                                 lr_decay, accuracy, val_accuracy, lr_final):
    """
    Tests learning rate options
    """
    framework = 'tensorflow'
    use_case = 'image_classification'
    output_dir = tempfile.mkdtemp()

    # Get the dataset
    dataset = dataset_factory.get_dataset('/tmp/data', 'image_classification', framework, dataset_name,
                                          'tf_datasets', split=["train[:5%]"])

    # Get the model
    model = model_factory.get_model(model_name, framework)
    model.learning_rate = learning_rate
    assert model.learning_rate == learning_rate

    # Preprocess the dataset
    dataset.preprocess(model.image_size, 32)
    dataset.shuffle_split(seed=10)

    # Train
    history = model.train(dataset, output_dir=output_dir, epochs=epochs, shuffle_files=False, seed=10, do_eval=do_eval,
                          early_stopping=early_stopping, lr_decay=lr_decay)

    assert history is not None
    assert history['acc'][-1] == accuracy
    if val_accuracy:
        assert history['val_acc'][-1] == val_accuracy
    else:
        assert 'val_acc' not in history
    if do_eval and lr_decay:
        assert history['lr'][-1] <= np.float32(lr_final)
    else:
        assert 'lr' not in history

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
