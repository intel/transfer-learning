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

from tlt.utils.file_utils import validate_model_name
from tlt.datasets import dataset_factory
from tlt.models import model_factory


@pytest.mark.integration
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
        history = model.train(dataset, output_dir=output_dir, epochs=1,
                              shuffle_files=False, do_eval=False,
                              extra_layers=extra_layers)
        assert history is not None
        assert len(model._model.layers) == correct_num_layers

        # Verify that checkpoints were generated
        cleaned_name = validate_model_name(model_name)
        checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(cleaned_name))
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

        # Retrain from checkpoints and verify that accuracy metric is the expected type
        retrain_model = model_factory.load_model(model_name, saved_model_dir, framework, 'text_classification')
        retrain_model.train(dataset, output_dir=output_dir, epochs=1, initial_checkpoints=checkpoint_dir,
                            shuffle_files=False, do_eval=False)

        retrain_metrics = retrain_model.evaluate(dataset)
        accuracy_index = next(id for id, k in enumerate(model._model.metrics_names) if 'acc' in k)
        # BERT model results are not deterministic, so the commented assertion doesn't reliably pass
        # assert retrain_metrics[accuracy_index] > trained_metrics[accuracy_index]
        assert isinstance(retrain_metrics[accuracy_index], float)

        # Test generating an Intel Neural Compressor config file (not implemented yet)
        inc_config_file_path = os.path.join(output_dir, "tf_{}.yaml".format(model_name))
        with pytest.raises(NotImplementedError):
            model.write_inc_config_file(inc_config_file_path, dataset, batch_size=batch_size,
                                        tuning_workspace=output_dir)

    finally:
        # Delete the temp output directory
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)


@pytest.mark.integration
@pytest.mark.tensorflow
@pytest.mark.parametrize('model_name, dataset_name, epochs, learning_rate, do_eval, \
                         lr_decay, accuracy, val_accuracy, lr_final',
                         [['small_bert/bert_en_uncased_L-2_H-128_A-2', 'glue/sst2', 1,
                           .005, False, False, None, None, 0.005],
                          ['small_bert/bert_en_uncased_L-2_H-256_A-4', 'glue/sst2',
                           1, .001, True, True, 0.34375, 0.4256, 0.001],
                          ['small_bert/bert_en_uncased_L-2_H-128_A-2', 'imdb_reviews',
                           15, .005, True, True, None, None, 0.001]])
def test_tf_binary_text_classification_with_lr_options(model_name, dataset_name,
                                                       epochs, learning_rate, do_eval,
                                                       lr_decay, accuracy, val_accuracy, lr_final):
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

        # TODO: BERT model results are not deterministic (AIZOO-1222), exact assertions will not pass
        # assert history['binary_accuracy'][-1] == accuracy
        # if val_accuracy:
        #     assert history['val_binary_accuracy'][-1] == val_accuracy
        # else:
        #     assert 'val_binary_accuracy' not in history

        # Non-determinism causes this assertion to fail a small fraction of the time,
        # for now, no assertions will be checked until a workaround is implemented
        if do_eval and lr_decay:
            pass
        else:
            assert 'lr' not in history

    finally:
        # Delete the temp output directory
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)


@pytest.mark.integration
@pytest.mark.tensorflow
@pytest.mark.parametrize('model_name',
                         ['small_bert/bert_en_uncased_L-2_H-128_A-2'])
def test_custom_dataset_workflow(model_name):
    """
    Tests the full workflow for TF text classification using a custom dataset
    """
    output_dir = tempfile.mkdtemp()

    def label_map_func(x):
        return int(x == "spam")

    try:
        # Get the dataset
        dataset = dataset_factory.load_dataset('/tmp/data/sms_spam_collection', use_case="text_classification",
                                               framework="tensorflow", csv_file_name="SMSSpamCollection",
                                               class_names=["ham", "spam"], shuffle_files=False,
                                               delimiter='\t', header=False, label_map_func=label_map_func)
        # Get the model
        model = model_factory.get_model(model_name, "tensorflow")

        # Preprocess the dataset and split to get small subsets for training and validation
        dataset.shuffle_split(train_pct=0.1, val_pct=0.1, shuffle_files=False)
        dataset.preprocess(batch_size=32)
        # Train for 1 epoch
        history = model.train(dataset=dataset, output_dir=output_dir, epochs=1, seed=10, do_eval=False)
        assert history is not None

        # Evaluate
        model.evaluate(dataset)

        # export the saved model
        saved_model_dir = model.export(output_dir)

        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb"))

        # Reload the saved model
        reload_model = model_factory.get_model(model_name, "tensorflow")
        reload_model.load_from_directory(saved_model_dir)

        # Evaluate
        metrics = reload_model.evaluate(dataset)
        assert len(metrics) > 0

        # Quantization
        inc_config_file_path = 'tlt/models/configs/inc/text_classification_template.yaml'
        nc_workspace = os.path.join(output_dir, "nc_workspace")
        model.write_inc_config_file(inc_config_file_path, dataset, batch_size=32, overwrite=True,
                                    accuracy_criterion_relative=0.1, exit_policy_max_trials=10,
                                    exit_policy_timeout=0, tuning_workspace=nc_workspace)
        quantization_output = os.path.join(output_dir, "quantized", "mocked")
        os.makedirs(quantization_output, exist_ok=True)
        model.quantize(saved_model_dir, quantization_output, inc_config_file_path)
        assert os.path.exists(os.path.join(quantization_output, "saved_model.pb"))

    finally:
        # Delete the temp output directory
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
