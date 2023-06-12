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
import numpy as np


from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.file_utils import download_and_extract_tar_file
from unittest.mock import MagicMock, patch

from tlt.datasets.image_classification.image_classification_dataset import ImageClassificationDataset

# This is necessary to protect from import errors when testing in a tensorflow only environment
keras_env = True

try:
    from tensorflow import keras
except ModuleNotFoundError:
    print("WARNING: Unable to import Keras. Tensorflow may not be installed")
    keras_env = False


@pytest.mark.integration
@pytest.mark.tensorflow
@pytest.mark.parametrize('model_name,dataset_name,train_accuracy,retrain_accuracy,extra_layers,correct_num_layers,'
                         'test_optimization',
                         [['efficientnet_b0', 'tf_flowers', 0.3125, 0.53125, None, 2, False],
                          ['resnet_v1_50', 'tf_flowers', 0.40625, 0.59375, None, 2, True],
                          ['efficientnet_b0', 'tf_flowers', 0.8125, 0.96875, [1024, 512], 4, False],
                          ['ResNet50', 'tf_flowers', 0.34375, 0.625, None, 4, True]])
def test_tf_image_classification(model_name, dataset_name, train_accuracy, retrain_accuracy, extra_layers,
                                 correct_num_layers, test_optimization):
    """
    Tests basic transfer learning functionality for TensorFlow image classification models using TF Datasets
    """
    framework = 'tensorflow'
    use_case = 'image_classification'
    output_dir = tempfile.mkdtemp()

    # Get the dataset
    dataset = dataset_factory.get_dataset('/tmp/data', use_case, framework, dataset_name,
                                          'tf_datasets', split=["train[:5%]"], seed=10, shuffle_files=False)

    # Get the model
    model = model_factory.get_model(model_name, framework)

    # Preprocess the dataset
    dataset.preprocess(model.image_size, 32, preprocessor=model.preprocessor)
    dataset.shuffle_split(shuffle_files=False)

    # Evaluate before training
    pretrained_metrics = model.evaluate(dataset)
    assert len(pretrained_metrics) > 0

    # Train
    history = model.train(dataset, output_dir=output_dir, epochs=1, shuffle_files=False, seed=10, do_eval=False,
                          extra_layers=extra_layers)
    assert history is not None
    np.testing.assert_almost_equal(history['acc'], [train_accuracy])
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
    probabilities = model.predict(images, return_type='probabilities')
    assert probabilities.shape == (32, 5)  # tf_flowers has 5 classes
    np.testing.assert_almost_equal(np.sum(probabilities), np.float32(32), decimal=4)

    # Export the saved model
    saved_model_dir = model.export(output_dir)
    assert os.path.isdir(saved_model_dir)
    assert os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb"))

    # Reload the saved model
    reload_model = model_factory.get_model(model_name, framework)
    reload_model.load_from_directory(saved_model_dir)

    # Evaluate
    reload_metrics = reload_model.evaluate(dataset)
    np.testing.assert_almost_equal(reload_metrics, trained_metrics)

    # Optimize the graph
    if test_optimization:
        inc_output_dir = os.path.join(output_dir, "optimized")
        os.makedirs(inc_output_dir, exist_ok=True)
        model.optimize_graph(inc_output_dir)
        assert os.path.isfile(os.path.join(inc_output_dir, "saved_model.pb"))

    # Retrain from checkpoints and verify that we have better accuracy than the original training
    retrain_model = model_factory.load_model(model_name, saved_model_dir, framework, use_case)
    retrain_history = retrain_model.train(dataset, output_dir=output_dir, epochs=1, initial_checkpoints=checkpoint_dir,
                                          shuffle_files=False, seed=10, do_eval=False)
    np.testing.assert_almost_equal(retrain_history['acc'], [retrain_accuracy])

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)


# This is necessary to protect from import errors when testing in a tensorflow only environment
if keras_env:
    @pytest.mark.integration
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
        dataset = dataset_factory.get_dataset('/tmp/data', use_case, framework, 'tf_flowers',
                                              'tf_datasets', split=["train[:5%]"], shuffle_files=False)

        # Define a custom model
        alexnet = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                                input_shape=(image_size, image_size, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
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
        probabilities = model.predict(images, return_type='probabilities')
        assert probabilities.shape == (32, 5)  # tf_flowers has 5 classes
        np.testing.assert_almost_equal(np.sum(probabilities), np.float32(32), decimal=4)

        # Export the saved model
        saved_model_dir = model.export(output_dir)
        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb"))

        # Reload the saved model
        reload_model = model_factory.load_model(model_name, saved_model_dir, framework, use_case)

        # Evaluate
        reload_metrics = reload_model.evaluate(dataset)
        np.testing.assert_almost_equal(reload_metrics, trained_metrics)

        # Retrain from checkpoints and verify that we have better accuracy than the original training
        retrain_model = model_factory.load_model(model_name, saved_model_dir, framework, use_case)
        retrain_history = retrain_model.train(dataset, output_dir=output_dir, epochs=1,
                                              initial_checkpoints=checkpoint_dir, shuffle_files=False, seed=10)
        assert retrain_history is not None

        # Delete the temp output directory
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)


@pytest.mark.integration
@pytest.mark.tensorflow
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

    @pytest.mark.parametrize('model_name,train_accuracy,retrain_accuracy,test_inc',
                             [['efficientnet_b0', 0.9333333, 1.0, False],
                              ['resnet_v1_50', 1.0, 1.0, True],
                              ['resnet_v2_50', 1.0, 1.0, False]])
    def test_custom_dataset_workflow(self, model_name, train_accuracy, retrain_accuracy, test_inc):
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
        dataset.shuffle_split(train_pct=0.1, val_pct=0.1, shuffle_files=False)
        dataset.preprocess(model.image_size, 32, preprocessor=model.preprocessor)

        # Train for 1 epoch
        history = model.train(dataset, output_dir=self._output_dir, epochs=1, shuffle_files=False, seed=10,
                              do_eval=False)
        assert history is not None
        np.testing.assert_almost_equal(history['acc'], [train_accuracy])

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
        np.testing.assert_almost_equal(retrain_history['acc'], [retrain_accuracy])

        # Test benchmarking, quantization
        if test_inc:
            inc_output_dir = os.path.join(self._output_dir, "quantized", model_name)
            os.makedirs(inc_output_dir)
            model.quantize(inc_output_dir, dataset=dataset)
            assert os.path.exists(os.path.join(inc_output_dir, "saved_model.pb"))
            model.benchmark(saved_model_dir=inc_output_dir, dataset=dataset)


@pytest.mark.integration
@pytest.mark.tensorflow
@pytest.mark.parametrize('model_name,dataset_name,epochs,learning_rate,do_eval,early_stopping,lr_decay,accuracy,\
                          val_accuracy,lr_final',
                         [['efficientnet_b0', 'tf_flowers', 4, 0.001, False, False, False, 0.9, None, 0.001],
                          ['efficientnet_b0', 'tf_flowers', 4, 0.001, True, False, False, 0.9, 0.8478260, 0.001],
                          ['efficientnet_b0', 'tf_flowers', 4, 0.001, True, False, True, 0.9, 0.8478260, 0.001],
                          ['efficientnet_b0', 'tf_flowers', 4, 0.001, False, False, True, 0.9, None, 0.001],
                          ['efficientnet_b0', 'tf_flowers', 16, 0.001, True, False, True, 1.0, 0.8695651, 1.0000e-03],
                          ['efficientnet_b0', 'tf_flowers', 25, 0.001, True, True, False, 1.0, 0.8695651, 0.0002]])
def test_tf_image_classification_with_lr_options(model_name, dataset_name, epochs, learning_rate, do_eval,
                                                 early_stopping, lr_decay, accuracy, val_accuracy, lr_final):
    """
    Tests learning rate options
    """
    framework = 'tensorflow'
    use_case = 'image_classification'
    output_dir = tempfile.mkdtemp()

    # Get the dataset
    dataset = dataset_factory.get_dataset('/tmp/data', use_case, framework, dataset_name,
                                          'tf_datasets', split=["train[:5%]"], shuffle_files=False)

    # Get the model
    model = model_factory.get_model(model_name, framework)
    model.learning_rate = learning_rate
    assert model.learning_rate == learning_rate

    # Preprocess the dataset
    dataset.shuffle_split(shuffle_files=False)
    dataset.preprocess(model.image_size, 32)

    # Train
    history = model.train(dataset, output_dir=output_dir, epochs=epochs, shuffle_files=False, seed=10, do_eval=do_eval,
                          early_stopping=early_stopping, lr_decay=lr_decay)

    assert history is not None
    np.testing.assert_almost_equal(history['acc'][-1], accuracy)
    if val_accuracy:
        np.testing.assert_almost_equal(history['val_acc'][-1], val_accuracy)
    else:
        assert 'val_acc' not in history
    if do_eval and lr_decay:
        assert history['lr'][-1] <= np.float32(lr_final)
    else:
        assert 'lr' not in history

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)


@pytest.mark.tensorflow
@pytest.mark.parametrize('add_aug',
                         ['rotate',
                          'zoom',
                          'hflip'])
def test_train_add_aug_mock(add_aug):
    """
    Tests basic add augmentation functionality for TensorFlow image classification models using mock objects
    """
    model = model_factory.get_model('efficientnet_b0', 'tensorflow')

    with patch('tlt.models.image_classification.tfhub_image_classification_model.'
               'TFHubImageClassificationModel._get_hub_model') as mock_get_hub_model:
        mock_dataset = MagicMock()
        mock_dataset.__class__ = ImageClassificationDataset
        print(mock_dataset.__class__)
        mock_dataset.validation_subset = [1, 2, 3]

        mock_dataset.class_names = ['a', 'b', 'c']
        mock_model = MagicMock()
        expected_return_value = {"result": True}
        mock_history = MagicMock()
        mock_history.history = expected_return_value

        def mock_fit(dataset, epochs, shuffle, callbacks, validation_data=None):
            assert dataset is not None
            assert isinstance(epochs, int)
            assert isinstance(shuffle, bool)
            assert len(callbacks) > 0

            return mock_history

        mock_model.fit = mock_fit
        mock_get_hub_model.return_value = mock_model

        # add basic preprocessing with add aug set to 'zoom'
        mock_dataset.preprocess(model.image_size, 32, add_aug=[add_aug])
        mock_dataset.shuffle_split(shuffle_files=False)

        # Test train without eval
        return_val = model.train(mock_dataset, output_dir="/tmp/output", do_eval=False)
        assert return_val == expected_return_value


@pytest.mark.tensorflow
def test_custom_callback():
    """
    Tests passing custom callbacks to the TensorFlow image classification train, evaluate, and predict functions.
    """
    model = model_factory.get_model('efficientnet_b0', 'tensorflow')

    with patch('tlt.models.image_classification.tfhub_image_classification_model.'
               'TFHubImageClassificationModel._get_hub_model') as mock_get_hub_model:
        mock_dataset = MagicMock()
        mock_dataset.__class__ = ImageClassificationDataset
        mock_dataset.validation_subset = [1, 2, 3]

        mock_dataset.class_names = ['a', 'b', 'c']
        mock_model = MagicMock()
        expected_return_value = {"result": True}
        mock_history = MagicMock()
        mock_history.history = expected_return_value

        class TestCallbackMethod(keras.callbacks.Callback):
            pass

        test_callback = TestCallbackMethod()

        def mock_fit(dataset, epochs, shuffle, callbacks, validation_data=None):
            # We should have more than one callback since TLT them and we added a custom one
            assert isinstance(callbacks, list)
            assert len(callbacks) > 1

            # We should have one callback that's our test callback
            assert (len([x for x in callbacks if x.__class__.__name__ == 'TestCallbackMethod']) == 1)

            return mock_history

        def mock_evaluate(dataset, callbacks=None):
            assert isinstance(callbacks, list)
            assert len(callbacks) == 1
            assert (len([x for x in callbacks if x.__class__.__name__ == 'TestCallbackMethod']) == 1)
            return [.98, 0.13]

        def mock_predict(input_samples, callbacks=None):
            assert isinstance(callbacks, list)
            assert len(callbacks) == 1
            assert (len([x for x in callbacks if x.__class__.__name__ == 'TestCallbackMethod']) == 1)
            return [1.0, 0.5]

        mock_model.fit.side_effect = mock_fit
        mock_model.evaluate.side_effect = mock_evaluate
        mock_model.predict.side_effect = mock_predict
        mock_get_hub_model.return_value = mock_model

        # Test custom callback as a single item, list, or tuple
        custom_callbacks = [test_callback, [test_callback], (test_callback)]

        for custom_callback in custom_callbacks:
            # Test train with custom callback
            return_val = model.train(mock_dataset, output_dir="/tmp/output", do_eval=False, callbacks=custom_callback)
            assert return_val == expected_return_value
            mock_model.fit.assert_called_once()
            mock_model.fit.reset_mock()

            # Test evaluate with custom callback
            model.evaluate(mock_dataset, callbacks=custom_callback)
            mock_model.evaluate.assert_called_once()
            mock_model.evaluate.reset_mock()

            # Test predict with custom callback
            model.predict([], callbacks=custom_callback)
            mock_model.predict.assert_called_once()
            mock_model.predict.reset_mock()


@pytest.mark.tensorflow
@patch('tlt.models.image_classification.tfhub_image_classification_model.TFHubImageClassificationModel._get_hub_model')
def test_invalid_callback_types(mock_get_hub_model):
    """
    Tests passing custom callbacks of the wrong type to train, predict, and evaluate
    """
    model = model_factory.get_model('efficientnet_b0', 'tensorflow')

    mock_dataset = MagicMock()
    mock_dataset.__class__ = ImageClassificationDataset
    mock_dataset.validation_subset = [1, 2, 3]

    mock_dataset.class_names = ['a', 'b', 'c']
    mock_model = MagicMock()
    expected_return_value = {"result": True}
    mock_history = MagicMock()
    mock_history.history = expected_return_value

    class TestCallbackMethod(keras.callbacks.Callback):
        pass

    good_callback = TestCallbackMethod()
    bad_callback = 1

    mock_model.fit = MagicMock()
    mock_model.evaluate = MagicMock()
    mock_model.predict = MagicMock()
    mock_get_hub_model.return_value = mock_model

    with pytest.raises(TypeError, match="Callbacks must be tf.keras.callbacks.Callback instances"):
        model.train(mock_dataset, output_dir="/tmp/output", do_eval=False, callbacks=[good_callback, bad_callback])

    with pytest.raises(TypeError, match="Callbacks must be tf.keras.callbacks.Callback instances"):
        model.evaluate(mock_dataset, callbacks=[good_callback, bad_callback])

    with pytest.raises(TypeError, match="Callbacks must be tf.keras.callbacks.Callback instances"):
        model.predict([], callbacks=[good_callback, bad_callback])
