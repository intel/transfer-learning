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
from unittest.mock import MagicMock

from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.datasets.text_classification.hf_custom_text_classification_dataset import HFCustomTextClassificationDataset


@pytest.mark.integration
@pytest.mark.pytorch
@pytest.mark.parametrize('model_name,dataset_name,extra_layers,correct_num_layers',
                         [['bert-base-cased', 'imdb', None, 1],
                          ['distilbert-base-uncased', 'imdb', [384, 192], 5]])
def test_pyt_text_classification(model_name, dataset_name, extra_layers, correct_num_layers):
    """
    Tests basic transfer learning functionality for PyTorch text classification models using a hugging face dataset
    """
    framework = 'pytorch'
    output_dir = tempfile.mkdtemp()

    # Get the dataset
    dataset = dataset_factory.get_dataset(output_dir, 'text_classification', framework, dataset_name,
                                          'huggingface', split=["train"], shuffle_files=False)

    # Get the model
    model = model_factory.get_model(model_name, framework)

    # Preprocess the dataset
    dataset.preprocess(model_name, batch_size=32)
    dataset.shuffle_split(train_pct=0.01, val_pct=0.01, seed=10)
    assert dataset._validation_type == 'shuffle_split'

    # Evaluate before training
    pretrained_metrics = model.evaluate(dataset)
    assert len(pretrained_metrics) > 0

    # Train
    model.train(dataset, output_dir=output_dir, epochs=1, do_eval=False, extra_layers=extra_layers)
    classifier_layer = getattr(model._model, "classifier")
    try:
        # If extra_layers given, the classifier is a Sequential layer with given input
        n_layers = len(classifier_layer)
    except TypeError:
        # If not given, the classifer is just a single Linear layer
        n_layers = 1
    assert n_layers == correct_num_layers

    # Evaluate
    trained_metrics = model.evaluate(dataset)
    assert trained_metrics[0] <= pretrained_metrics[0]  # loss
    assert trained_metrics[1] >= pretrained_metrics[1]  # accuracy

    # Export the saved model
    saved_model_dir = model.export(output_dir)
    assert os.path.isdir(saved_model_dir)
    assert os.path.isfile(os.path.join(saved_model_dir, "model.pt"))

    # Reload the saved model
    reload_model = model_factory.get_model(model_name, framework)
    reload_model.load_from_directory(saved_model_dir, num_classes=len(dataset.class_names))

    # Evaluate
    reload_metrics = reload_model.evaluate(dataset)
    assert reload_metrics == trained_metrics

    # Ensure we get 'NotImplementedError' for graph_optimization
    with pytest.raises(NotImplementedError):
        model.optimize_graph(saved_model_dir, os.path.join(saved_model_dir, 'optimized'))

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)


@pytest.mark.integration
@pytest.mark.pytorch
@pytest.mark.parametrize('model_name',
                         ['bert-base-cased'])
def test_custom_dataset_workflow(model_name):
    """
    Tests the full workflow for PYT text classification using a custom dataset mock
    """
    model = model_factory.get_model(model_name, framework='pytorch', use_case="text_classification")

    output_dir = tempfile.mkdtemp()
    os.environ["TORCH_HOME"] = output_dir

    mock_dataset = MagicMock()
    mock_dataset.__class__ = HFCustomTextClassificationDataset
    mock_dataset.validation_subset = ['fun', 'terrible']
    mock_dataset.train_subset = ["fun, happy, boring, terrible"]
    mock_dataset.class_names = ['good', 'bad']

    # Preprocess the dataset and split to get small subsets for training and validation
    mock_dataset.shuffle_split(train_pct=0.1, val_pct=0.1, shuffle_files=False)
    mock_dataset.preprocess(model_name, batch_size=32)

    # Train for 1 epoch
    history = model.train(mock_dataset, output_dir=output_dir, epochs=1, seed=10, do_eval=False)
    assert history is not None

    # Evaluate
    model.evaluate(mock_dataset)

    # export the saved model
    saved_model_dir = model.export(output_dir)
    assert os.path.isdir(saved_model_dir)
    assert os.path.isfile(os.path.join(saved_model_dir, "model.pt"))

    # Reload the saved model
    reload_model = model_factory.get_model(model_name, 'pytorch')
    reload_model.load_from_directory(saved_model_dir, 2)

    # Evaluate
    metrics = reload_model.evaluate(mock_dataset)
    assert len(metrics) > 0

    # Quantization
    inc_config_file_path = 'tlt/models/configs/inc/text_classification_template.yaml'
    nc_workspace = os.path.join(output_dir, "nc_workspace")
    model.write_inc_config_file(inc_config_file_path, mock_dataset, batch_size=32, overwrite=True,
                                accuracy_criterion_relative=0.1, exit_policy_max_trials=10,
                                exit_policy_timeout=0, tuning_workspace=nc_workspace)
    quantization_output = os.path.join(output_dir, "quantized", model_name)
    os.makedirs(quantization_output, exist_ok=True)
    model.quantize(saved_model_dir, quantization_output, inc_config_file_path)
    assert os.path.exists(os.path.join(quantization_output, "model.pt"))

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)


@pytest.mark.pytorch
@pytest.mark.parametrize('model_name,dataset_name',
                         [['distilbert-base-uncased', 'imdb']])
def test_initial_checkpoints(model_name, dataset_name):
    framework = 'pytorch'
    output_dir = tempfile.mkdtemp()
    checkpoint_dir = os.path.join(output_dir, model_name + '_checkpoints')

    # Get the dataset
    dataset = dataset_factory.get_dataset(output_dir, 'text_classification', framework, dataset_name,
                                          'huggingface', split=["train"], shuffle_files=False)

    # Get the model
    model = model_factory.get_model(model_name, framework)

    assert model._generate_checkpoints is True

    dataset.preprocess(model_name, batch_size=32)
    dataset.shuffle_split(train_pct=0.01, val_pct=0.01, seed=10)

    # Train
    model.train(dataset, output_dir=output_dir, epochs=2, do_eval=False)

    trained_metrics = model.evaluate(dataset)

    # Delete the model and train a brand new model but instead we resume training from checkpoints
    del model

    model = model_factory.get_model(model_name, framework)
    model.train(dataset, output_dir=output_dir, epochs=2, do_eval=False,
                initial_checkpoints=os.path.join(checkpoint_dir, 'checkpoint.pt'))

    improved_metrics = model.evaluate(dataset)

    assert improved_metrics[0] < trained_metrics[0]  # loss
    assert improved_metrics[1] > trained_metrics[1]  # accuracy

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)


@pytest.mark.pytorch
@pytest.mark.parametrize('model_name,dataset_name',
                         [['distilbert-base-uncased', 'imdb']])
def test_freeze_bert(model_name, dataset_name):
    framework = 'pytorch'
    output_dir = tempfile.mkdtemp()

    # Get the dataset
    dataset = dataset_factory.get_dataset(output_dir, 'text_classification', framework, dataset_name,
                                          'huggingface', split=["train"], shuffle_files=False)

    # Get the model
    model = model_factory.get_model(model_name, framework)

    dataset.preprocess(model_name, batch_size=32)
    dataset.shuffle_split(train_pct=0.01, val_pct=0.01, seed=10)

    # Train
    model.train(dataset, output_dir=output_dir, epochs=1, do_eval=False)

    # Freeze feature layers
    layer_name = "features"
    model.freeze_layer(layer_name)

    # Check everything is frozen (not trainable) in the layer
    for (name, module) in model._model.named_children():
        if name == layer_name:
            for param in module.parameters():
                assert param.requires_grad is False

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)


@pytest.mark.pytorch
@pytest.mark.parametrize('model_name,dataset_name',
                         [['distilbert-base-uncased', 'imdb']])
def test_unfreeze_bert(model_name, dataset_name):
    framework = 'pytorch'
    output_dir = tempfile.mkdtemp()

    # Get the dataset
    dataset = dataset_factory.get_dataset(output_dir, 'text_classification', framework, dataset_name,
                                          'huggingface', split=["train"], shuffle_files=False)

    # Get the model
    model = model_factory.get_model(model_name, framework)

    dataset.preprocess(model_name, batch_size=32)
    dataset.shuffle_split(train_pct=0.01, val_pct=0.01, seed=10)

    # Train
    model.train(dataset, output_dir=output_dir, epochs=1, do_eval=False)
    layer_name = "features"
    model.unfreeze_layer(layer_name)

    # Check everything is unfrozen (trainable) in the layer
    for (name, module) in model._model.named_children():
        if name == layer_name:
            for param in module.parameters():
                assert param.requires_grad is True

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)


@pytest.mark.pytorch
@pytest.mark.parametrize('model_name,dataset_name',
                         [['distilbert-base-uncased', 'imdb']])
def test_list_layers_bert(model_name, dataset_name):
    import io
    import unittest.mock as mock

    framework = 'pytorch'
    output_dir = tempfile.mkdtemp()

    # Get the model
    model = model_factory.get_model(model_name, framework)

    # Get the dataset
    dataset = dataset_factory.get_dataset(output_dir, 'text_classification', framework, dataset_name,
                                          'huggingface', split=["train"], shuffle_files=False)

    dataset.preprocess(model_name, batch_size=32)
    dataset.shuffle_split(train_pct=0.01, val_pct=0.01, seed=10)

    # Train
    model.train(dataset, output_dir=output_dir, epochs=1, do_eval=False)

    # Mock stdout and sterr to capture the function's output
    stdout = io.StringIO()
    stderr = io.StringIO()
    with mock.patch('sys.stdout', stdout), mock.patch('sys.stderr', stderr):
        model.list_layers(verbose=True)
    # Assert the function printed the correct output of the trainable layers
    output = stdout.getvalue().strip()
    assert 'distilbert' in output
    assert 'embeddings: 23835648/23835648 parameters are trainable' in output
    assert 'transformer: 42527232/42527232 parameters are trainable' in output
    assert 'pre_classifier: 590592/590592 parameters are trainable' in output
    assert 'dropout: 0/0 parameters are trainable' in output
    assert 'dropout: 0/0 parameters are trainable' in output
    assert 'Total Trainable Parameters: 66955010/66955010' in output

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
