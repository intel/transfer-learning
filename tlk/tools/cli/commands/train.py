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
import sys


@click.command()
@click.option("--framework", "-f",
              required=False,
              default="tensorflow",
              help="Deep learning framework [default: tensorflow]")
@click.option("--model-name", "--model_name",
              required=True,
              type=str,
              help="Name of the model to use")
@click.option("--output-dir", "--output_dir",
              required=True,
              type=str,
              help="Output directory for saved models, logs, checkpoints, etc")
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
              help="Name of a dataset catalog for a named dataset (Options: "
                   "tf_datasets, torchvision, huggingface). If a dataset name is provided "
                   "and no dataset catalog is given, it will default to use tf_datasets for a TensorFlow "
                   "model, torchvision for PyTorch CV models, and huggingface datasets for HuggingFace models.")
@click.option("--epochs",
              default=1,
              type=int,
              help="Number of training epochs [default: 1]")
def train(framework, model_name, output_dir, dataset_dir, dataset_name, dataset_catalog, epochs):
    """
    Trains the model
    """
    print("Model name:", model_name)
    print("Framework:", framework)

    if dataset_name:
        print("Dataset name:", dataset_name)

        if dataset_catalog:
            print("Dataset catalog:", dataset_catalog)

    print("Training epochs:", epochs)
    print("Dataset dir:", dataset_dir)
    print("Output directory:", output_dir, flush=True)

    from tlk.models import model_factory
    from tlk.datasets import dataset_factory
    from tlk.datasets.image_classification.tf_image_classification_dataset import TFImageClassificationDataset

    # Get the model
    try:
        model = model_factory.get_model(model_name, framework)
    except Exception as e:
        sys.exit("Error while getting the model (model name: {}, framework: {}):\n{}".format(
            model_name, framework, str(e)))

    # Get the dataset
    try:
        dataset = dataset_factory.get_dataset(dataset_dir, model.use_case, model.framework, dataset_name, dataset_catalog)

        # TODO: get extra configs like batch size and maybe this doesn't need to be a separate call
        if isinstance(dataset, TFImageClassificationDataset):
            dataset.preprocess(model.image_size, batch_size=32)
    except Exception as e:
        sys.exit("Error while getting the dataset (dataset dir: {}, use case: {}, framework: {}, "
                 "dataset name: {}, dataset_catalog: {}):\n{}".format(
            dataset_dir, model.use_case,  model.framework, dataset_name, dataset_catalog, str(e)))

    # Train the model using the dataset
    try:
        model.train(dataset, output_dir=output_dir, epochs=epochs)
    except Exception as e:
        sys.exit("There was an error during model training:\n{}".format(str(e)))

    # Save the trained model
    try:
        # TODO: also export info about how the model was trained, logs, etc so that it's reproducible
        model.export(output_dir)
    except Exception as e:
        sys.exit("There was an error when saving the model:\n{}".format(str(e)))
