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

from tlt.utils.file_utils import validate_model_name, download_and_extract_zip_file
from tlt.datasets import dataset_factory
from tlt.models import model_factory


@pytest.mark.integration
@pytest.mark.tensorflow
@pytest.mark.parametrize('model_name,dataset_name,extra_layers,correct_num_layers,model_hub',
                         [['google/bert_uncased_L-2_H-128_A-2', 'ag_news_subset', None, 5, 'huggingface']])
def test_tf_multi_text_classification(model_name, dataset_name, extra_layers, correct_num_layers, model_hub):
    """
    Tests basic transfer learning functionality for TensorFlow multi text classification using TF Datasets
    """
    framework = 'tensorflow'
    output_dir = tempfile.mkdtemp()
    os.environ["TENSORFLOW_HOME"] = output_dir

    try:
        # Get the dataset
        dataset = dataset_factory.get_dataset(output_dir, 'text_classification', framework, dataset_name,
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

        text1 = ('Oil and Economy Cloud Stocks Outlook (Reuters) Reuters - '
                 'Soaring crude prices plus worries about the economy and the'
                 'outlook for earnings are expected to hang over the stock market'
                 'next week during the depth of the summer doldrums')
        text2 = ('Wall St. Bears Claw Back Into the Black (Reuters) Reuters -'
                 'Short-sellers, Wall Streets dwindlingband of ultra-cynics,'
                 'are seeing green again.')
        text3 = ('Expansion slows in Japan Economic growth in Japan slows down'
                 'as the country experiences a drop in domestic and corporate spending.'
                 'outlook for earnings are expected to hang over the stock market'
                 'next week during the depth of the summer doldrums')
        # Predict with raw text input
        raw_text_input = [text1, text2, text3]
        predictions = model.predict(raw_text_input)
        assert len(predictions) == len(raw_text_input)

        # export the saved model
        saved_model_dir = model.export(output_dir)
        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb"))

        # Reload the saved model
        reload_model = model_factory.load_model(model_name, saved_model_dir, framework, 'text_classification',
                                                model_hub)

        # Evaluate
        reload_metrics = reload_model.evaluate(dataset)
        assert reload_metrics == trained_metrics

        # Predict with the raw text input
        reload_predictions = reload_model.predict(raw_text_input)
        assert (reload_predictions == predictions).all()

        # Retrain from checkpoints and verify that accuracy metric is the expected type
        retrain_model = model_factory.load_model(model_name, saved_model_dir, framework, 'text_classification',
                                                 model_hub)
        retrain_model.train(dataset, output_dir=output_dir, epochs=1, initial_checkpoints=checkpoint_dir,
                            shuffle_files=False, do_eval=False)

        retrain_metrics = retrain_model.evaluate(dataset)
        accuracy_index = next(id for id, k in enumerate(model._model.metrics_names) if 'acc' in k)
        # BERT model results are not deterministic, so the commented assertion doesn't reliably pass
        assert isinstance(retrain_metrics[accuracy_index], float)

    finally:
        # Delete the temp output directory
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)


@pytest.mark.integration
@pytest.mark.tensorflow
@pytest.mark.parametrize('model_name,dataset_name,extra_layers,correct_num_layers,model_hub',
                         [['google/bert_uncased_L-2_H-128_A-2', 'imdb_reviews', None, 5, 'huggingface'],
                          ['google/bert_uncased_L-2_H-256_A-4', 'glue/sst2', None, 5, 'huggingface'],
                          ['google/bert_uncased_L-2_H-128_A-2', 'imdb_reviews', [512, 128], 7, 'huggingface']])
def test_tf_binary_text_classification(model_name, dataset_name, extra_layers, correct_num_layers, model_hub):
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
        reload_model = model_factory.load_model(model_name, saved_model_dir, framework, 'text_classification',
                                                model_hub)

        # Evaluate
        reload_metrics = reload_model.evaluate(dataset)
        assert reload_metrics == trained_metrics

        # Predict with the raw text input
        reload_predictions = reload_model.predict(raw_text_input)
        assert (reload_predictions == predictions).all()

        # Retrain from checkpoints and verify that accuracy metric is the expected type
        retrain_model = model_factory.load_model(model_name, saved_model_dir, framework, 'text_classification',
                                                 model_hub)
        retrain_model.train(dataset, output_dir=output_dir, epochs=1, initial_checkpoints=checkpoint_dir,
                            shuffle_files=False, do_eval=False)

        retrain_metrics = retrain_model.evaluate(dataset)
        accuracy_index = next(id for id, k in enumerate(model._model.metrics_names) if 'acc' in k)
        # BERT model results are not deterministic, so the commented assertion doesn't reliably pass
        # assert retrain_metrics[accuracy_index] > trained_metrics[accuracy_index]
        assert isinstance(retrain_metrics[accuracy_index], float)

    finally:
        # Delete the temp output directory
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)


@pytest.mark.integration
@pytest.mark.tensorflow
@pytest.mark.parametrize('model_name, dataset_name, epochs, learning_rate, do_eval, \
                         lr_decay, accuracy, val_accuracy, lr_final',
                         [['google/bert_uncased_L-2_H-128_A-2', 'glue/sst2', 1,
                           .005, False, False, None, None, 0.005],
                          ['google/bert_uncased_L-2_H-256_A-4', 'glue/sst2',
                           1, .001, True, True, 0.34375, 0.4256, 0.001],
                          ['google/bert_uncased_L-2_H-128_A-2', 'imdb_reviews',
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
                         ['google/bert_uncased_L-2_H-128_A-2'])
def test_custom_dataset_workflow(model_name):
    """
    Tests the full workflow for TF text classification using a custom dataset
    """
    output_dir = tempfile.mkdtemp()
    dataset_dir = '/tmp/data'

    def label_map_func(x):
        return int(x == "spam")

    try:
        # Get the dataset
        zip_file_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        sms_data_directory = os.path.join(dataset_dir, "sms_spam_collection")
        csv_file_name = "SMSSpamCollection"

        # If the SMS Spam collection csv file is not found, download and extract the file:
        if not os.path.exists(os.path.join(sms_data_directory, csv_file_name)):
            # Download the zip file with the SMS Spam collection dataset
            download_and_extract_zip_file(zip_file_url, sms_data_directory)

        dataset = dataset_factory.load_dataset(sms_data_directory, use_case="text_classification",
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
        inc_output_dir = os.path.join(output_dir, "quantized", "mocked")
        os.makedirs(inc_output_dir, exist_ok=True)
        model.quantize(inc_output_dir, dataset)
        assert os.path.exists(os.path.join(inc_output_dir, "saved_model.pb"))

    finally:
        # Delete the temp output directory
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
