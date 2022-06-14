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

import click
import os
import sys

from tlk.utils.types import FrameworkType


@click.command()
@click.option("--model-dir", "--model_dir",
              required=True,
              type=str,
              help="Model directory to reload and evaluate a model exported by TLK.")
@click.option("--dataset-dir", "--dataset_dir",
              required=True,
              type=str,
              help="Dataset directory for a custom dataset, or if a dataset name "
                   "and catalog are being provided, the dataset directory is the "
                   "location where the dataset will be downloaded.")
@click.option("--dataset-name", "--dataset_name",
              required=False,
              type=str,
              help="Name of the dataset to use from a dataset catalog.")
@click.option("--dataset-catalog", "--dataset_catalog",
              required=False,
              type=str,
              help="Name of a dataset catalog for a named dataset (Options: tf_datasets, torchvision, huggingface). "
                   "If a dataset name is provided and no dataset catalog is given, it will default to use "
                   "tf_datasets for a TensorFlow model, torchvision for PyTorch CV models, and huggingface datasets "
                   "for HuggingFace models.")
def eval(model_dir, dataset_dir, dataset_name, dataset_catalog):
    """
    Evaluates a model that has already been trained
    """
    print("Model directory:", model_dir)
    print("Dataset directory:", dataset_dir)

    if dataset_name:
        print("Dataset name:", dataset_name)
        if dataset_catalog:
            print("Dataset catalog:", dataset_catalog)

    try:
        from tlk.utils.file_utils import verify_directory
        verify_directory(model_dir, require_directory_exists=True)
    except Exception as e:
        sys.exit("Error while verifying the model directory: {}", str(e))

    saved_model_path = os.path.join(model_dir, "saved_model.pb")
    pytorch_model_path = os.path.join(model_dir, "model.pt")
    if os.path.isfile(saved_model_path):
        framework = FrameworkType.TENSORFLOW
        model_path = saved_model_path
    elif os.path.isfile(pytorch_model_path):
        framework = FrameworkType.PYTORCH
        model_path = pytorch_model_path
    else:
        sys.exit("Evaluation is currently only implemented for TensorFlow saved models and PyTorch .pt models. No such files "
                 "found in the model directory ({}).".format(model_dir))
    model_name = os.path.basename(os.path.dirname(model_dir))

    print("Model name:", model_name)
    print("Framework:", framework)

    try:
        from tlk.models.model_factory import get_model

        print("Loading model object for {} using {}".format(model_name, str(framework)), flush=True)
        model = get_model(model_name, framework)

        print("Loading saved model from:", model_path)
        model.load_from_directory(model_dir)

        from tlk.datasets import dataset_factory
        from tlk.datasets.image_classification.image_classification_dataset import ImageClassificationDataset

        if not dataset_catalog and not dataset_name:
            dataset = dataset_factory.load_dataset(dataset_dir, model.use_case, model.framework)
        else:
            dataset = dataset_factory.get_dataset(dataset_dir, model.use_case, model.framework, dataset_name, dataset_catalog)

        if isinstance(dataset, ImageClassificationDataset):
            dataset.preprocess(model.image_size, batch_size=32)
            dataset.shuffle_split(seed=10)

        model.evaluate(dataset)
    except Exception as e:
        sys.exit("An error occurred during evaluation: {}".format(str(e)))
