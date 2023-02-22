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

import click
import inspect
import os
import sys

from tlt.utils.types import FrameworkType


@click.command()
@click.option("--model-dir", "--model_dir",
              required=True,
              type=str,
              help="Model directory to reload and evaluate a previously exported model.")
@click.option("--model-name", "--model_name",
              required=False,
              type=str,
              help="Name of the model to evaluate. If a model name is not provided, the CLI will try to get the model "
                   "name from the model directory path. For example, if the model directory is /tmp/efficientnet_b0/10,"
                   " it will use 'efficientnet_b0' as the model name.")
@click.option("--dataset-dir", "--dataset_dir",
              required=True,
              type=str,
              help="Dataset directory for a custom dataset, or if a dataset name "
                   "and catalog are being provided, the dataset directory is the "
                   "location where the dataset will be downloaded.")
@click.option("--dataset-file", "--dataset_file",
              required=False,
              type=str,
              help="Name of a file in the dataset directory to load. Used for loading a .csv file for text "
                   "classification evaluation.")
@click.option("--delimiter",
              required=False,
              type=str,
              default=",",
              help="Delimiter used when loading a dataset from a csv file. [default: ,]")
@click.option("--class-names", "--class_names",
              required=False,
              type=str,
              help="Comma separated string of class names for a text classification dataset being loaded from .csv")
@click.option("--dataset-name", "--dataset_name",
              required=False,
              type=str,
              help="Name of the dataset to use from a dataset catalog.")
@click.option("--dataset-catalog", "--dataset_catalog",
              required=False,
              type=click.Choice(['tf_datasets', 'torchvision', 'huggingface']),
              help="Name of a dataset catalog for a named dataset (Options: tf_datasets, torchvision, huggingface). "
                   "If a dataset name is provided and no dataset catalog is given, it will default to use "
                   "tf_datasets for a TensorFlow model, torchvision for PyTorch CV models, and huggingface datasets "
                   "for HuggingFace models.")
def eval(model_dir, model_name, dataset_dir, dataset_file, delimiter, class_names, dataset_name, dataset_catalog):
    """
    Evaluates a model that has already been trained
    """
    print("Model directory:", model_dir)
    print("Dataset directory:", dataset_dir)

    if dataset_file:
        print("Dataset file:", dataset_file)

    if class_names:
        class_names = class_names.split(",")
        print("Class names:", class_names)

    if dataset_name:
        print("Dataset name:", dataset_name)
        if dataset_catalog:
            print("Dataset catalog:", dataset_catalog)

    try:
        from tlt.utils.file_utils import verify_directory
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
        sys.exit("Evaluation is currently only implemented for TensorFlow saved models and PyTorch .pt models. No such "
                 "files found in the model directory ({}).".format(model_dir))

    if not model_name:
        model_name = os.path.basename(os.path.dirname(model_dir))

    print("Model name:", model_name)
    print("Framework:", framework)

    try:
        from tlt.models.model_factory import get_model

        print("Loading model object for {} using {}".format(model_name, str(framework)), flush=True)
        model = get_model(model_name, framework)

        print("Loading saved model from:", model_path)
        model.load_from_directory(model_dir)

        from tlt.datasets import dataset_factory

        if not dataset_catalog and not dataset_name:
            if str(model.use_case) == 'text_classification':
                if not dataset_file:
                    raise ValueError("Loading a text classification dataset requires --dataset-file to specify the "
                                     "file name of the .csv file to load from the --dataset-dir.")
                if not class_names:
                    raise ValueError("Loading a text classification dataset requires --class-names to specify a list "
                                     "of the class labels for the dataset.")

                dataset = dataset_factory.load_dataset(dataset_dir, model.use_case, model.framework, dataset_name,
                                                       class_names=class_names, csv_file_name=dataset_file,
                                                       delimiter=delimiter)
            else:
                dataset = dataset_factory.load_dataset(dataset_dir, model.use_case, model.framework)
        else:
            dataset = dataset_factory.get_dataset(dataset_dir, model.use_case, model.framework, dataset_name,
                                                  dataset_catalog)

        if 'image_size' in inspect.getfullargspec(dataset.preprocess).args:
            dataset.preprocess(image_size=model.image_size, batch_size=32)
        else:
            dataset.preprocess(batch_size=32)

        dataset.shuffle_split(seed=10)

        model.evaluate(dataset)
    except Exception as e:
        sys.exit("An error occurred during evaluation: {}".format(str(e)))
