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

from tlt.datasets import dataset_factory
from tlt.models import model_factory


@pytest.mark.tensorflow
@pytest.mark.parametrize('model_name,dataset_name,extra_layers,correct_num_layers',
                         [['small_bert/bert_en_uncased_L-2_H-128_A-2', 'imdb_reviews', None, 3],
                          ['small_bert/bert_en_uncased_L-2_H-256_A-4', 'glue/sst2', None, 3],
                          ['small_bert/bert_en_uncased_L-2_H-128_A-2', 'imdb_reviews', [512, 128], 5]])
def test_tf_binary_text_classification(model_name, dataset_name, extra_layers, correct_num_layers):
    """
    Tests basic transfer learning functionality for TensorFlow binary text classification using TF Datasets
    """
    framework = 'tensorflow'
    output_dir = tempfile.mkdtemp()

    try:
        # Get the dataset
        dataset = dataset_factory.get_dataset('/tmp/data', 'text_classification', framework, dataset_name,
                                              'tf_datasets', split=["train[:8%]"], shuffle_files=False)

        # Get the model
        model = model_factory.get_model(model_name, framework)

        # Preprocess the dataset
        batch_size = 32
        dataset.preprocess(batch_size)
        dataset.shuffle_split(seed=10)

        # This model does not support evaluate/predict before training
        with pytest.raises(ValueError) as e:
            model.evaluate(dataset)
        assert "model must be trained" in str(e)
        with pytest.raises(ValueError) as e:
            model.predict(dataset)
        assert "model must be trained" in str(e)

        # Train
        history = model.train(dataset, output_dir=output_dir, epochs=1, shuffle_files=False, do_eval=False, 
                              extra_layers=extra_layers)
        assert history is not None
        assert len(model._model.layers) == correct_num_layers

        # Verify that checkpoints were generated
        checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(model_name))
        assert os.path.isdir(checkpoint_dir)
        assert len(os.listdir(checkpoint_dir))

        # Evaluate
        trained_metrics = model.evaluate(dataset)
        assert len(trained_metrics) == 2  # expect to get loss and accuracy metrics

        # Predict with a batch
        input, labels = dataset.get_batch()
        predictions = model.predict(input)
        assert len(predictions) == batch_size

        # Predict with raw text input
        raw_text_input = ["awesome", "fun", "boring"]
        predictions = model.predict(raw_text_input)
        assert len(predictions) == len(raw_text_input)

        # export the saved model
        saved_model_dir = model.export(output_dir)
        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb"))

        # Reload the saved model
        reload_model = model_factory.load_model(model_name, saved_model_dir, framework, 'text_classification')

        # Evaluate
        reload_metrics = reload_model.evaluate(dataset)
        assert reload_metrics == trained_metrics

        # Predict with the raw text input
        reload_predictions = reload_model.predict(raw_text_input)
        assert (reload_predictions == predictions).all()

        # Retrain from checkpoints and verify that we have better accuracy than the original training
        retrain_model = model_factory.load_model(model_name, saved_model_dir, framework, 'text_classification')
        retrain_model.train(dataset, output_dir=output_dir, epochs=1, initial_checkpoints=checkpoint_dir,
                            shuffle_files=False, do_eval=False)
        retrain_metrics = retrain_model.evaluate(dataset)
        accuracy_index = next(id for id, k in enumerate(model._model.metrics_names) if 'acc' in k)
        assert retrain_metrics[accuracy_index] > trained_metrics[accuracy_index]

        # Test generating an INC config file (not implemented yet)
        inc_config_file_path = os.path.join(output_dir, "tf_{}.yaml".format(model_name))
        with pytest.raises(NotImplementedError):
            model.write_inc_config_file(inc_config_file_path, dataset, batch_size=batch_size,
                                        tuning_workspace=output_dir)

    finally:
        # Delete the temp output directory
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

@pytest.mark.tensorflow
@pytest.mark.parametrize('model_name,dataset_name,epochs,learning_rate,do_eval,lr_decay,accuracy,val_accuracy,lr_final',
                         [['small_bert/bert_en_uncased_L-2_H-128_A-2', 'glue/sst2', 1, .005, False, False, None, None, 0.005],
                          ['small_bert/bert_en_uncased_L-2_H-256_A-4', 'glue/sst2', 1, .001, True, True, 0.34375, 0.4256, 0.001],
                          ['small_bert/bert_en_uncased_L-2_H-128_A-2', 'imdb_reviews', 15, .005, True, True, None, None, 0.001]])
def test_tf_binary_text_classification_with_lr_options(model_name, dataset_name, epochs, learning_rate, do_eval, lr_decay,
                                                 accuracy, val_accuracy, lr_final):
    """
    Tests transfer learning for TensorFlow binary text classification with different learning rate options
    """
    framework = 'tensorflow'
    output_dir = tempfile.mkdtemp()

    try:
        # Get the dataset
        dataset = dataset_factory.get_dataset('/tmp/data', 'text_classification', framework, dataset_name,
                                              'tf_datasets', split=["train[:4%]"], shuffle_files=False)

        # Get the model
        model = model_factory.get_model(model_name, framework)
        model.learning_rate = learning_rate
        assert model.learning_rate == learning_rate

        # Preprocess the dataset
        batch_size = 32
        dataset.preprocess(batch_size)
        dataset.shuffle_split(seed=10)

        # Train
        history = model.train(dataset, output_dir=output_dir, epochs=epochs, shuffle_files=False, do_eval=do_eval,
                              lr_decay=lr_decay, seed=10)
        assert history is not None

        # TODO: accuracy results are not deterministic yet (AIZOO-1222)
        # assert history['binary_accuracy'][-1] == accuracy
        # if val_accuracy:
        #     assert history['val_binary_accuracy'][-1] == val_accuracy
        # else:
        #     assert 'val_binary_accuracy' not in history

        if do_eval and lr_decay:
            assert history['lr'][-1] <= np.float32(lr_final)
        else:
            assert 'lr' not in history

    finally:
        # Delete the temp output directory
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
