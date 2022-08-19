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
import inspect
import sys


@click.command()
@click.option("--framework", "-f",
              required=False,
              default="tensorflow",
              type=click.Choice(['tensorflow', 'pytorch']),
              help="Deep learning framework [default: tensorflow]")
@click.option("--model-name", "--model_name",
              required=True,
              type=str,
              help="Name of the model to use")
@click.option("--output-dir", "--output_dir",
              required=True,
              type=click.Path(dir_okay=True, file_okay=False),
              help="Output directory for saved models, logs, checkpoints, etc")
@click.option("--dataset-dir", "--dataset_dir",
              required=True,
              type=click.Path(dir_okay=True, file_okay=False),
              help="Dataset directory for a custom dataset, or if a dataset name "
                   "and catalog are being provided, the dataset directory is the "
                   "location where the dataset will be downloaded.")
@click.option("--dataset-file", "--dataset_file",
              required=False,
              type=str,
              help="Name of a file in the dataset directory to load. Used for loading a .csv file for text "
                   "classification fine tuning.")
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
              help="Name of a dataset catalog for a named dataset (Options: "
                   "tf_datasets, torchvision, huggingface). If a dataset name is provided "
                   "and no dataset catalog is given, it will default to use tf_datasets for a TensorFlow "
                   "model, torchvision for PyTorch CV models, and huggingface datasets for HuggingFace models.")
@click.option("--epochs",
              default=1,
              type=click.IntRange(min=1),
              help="Number of training epochs [default: 1]")
@click.option("--init-checkpoints", "--init_checkpoints",
              required=False,
              type=click.Path(dir_okay=True),
              help="Optional path to checkpoint weights to load to resume training. If the path provided is a "
                   "directory, the latest checkpoint from the directory will be used.")
def train(framework, model_name, output_dir, dataset_dir, dataset_file, delimiter, class_names, dataset_name,
          dataset_catalog, epochs, init_checkpoints):
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

    if init_checkpoints:
        print("Initial checkpoints:", init_checkpoints)

    print("Dataset dir:", dataset_dir)

    if dataset_file:
        print("Dataset file:", dataset_file)

    if class_names:
        class_names = class_names.split(",")
        print("Class names:", class_names)

    print("Output directory:", output_dir, flush=True)

    from tlt.models import model_factory
    from tlt.datasets import dataset_factory

    # Get the model
    try:
        model = model_factory.get_model(model_name, framework)
    except Exception as e:
        sys.exit("Error while getting the model (model name: {}, framework: {}):\n{}".format(
            model_name, framework, str(e)))

    # Get the dataset
    try:
        if not dataset_name and not dataset_catalog:
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
            dataset = dataset_factory.get_dataset(dataset_dir, model.use_case, model.framework, dataset_name, dataset_catalog)

        # TODO: get extra configs like batch size and maybe this doesn't need to be a separate call
        if framework in ['tensorflow', 'pytorch']:
            if 'image_size' in inspect.getfullargspec(dataset.preprocess).args:
                dataset.preprocess(image_size=model.image_size, batch_size=32)
            else:
                dataset.preprocess(batch_size=32)
            dataset.shuffle_split(seed=10)
    except Exception as e:
        sys.exit("Error while getting the dataset (dataset dir: {}, use case: {}, framework: {}, "
                 "dataset name: {}, dataset_catalog: {}):\n{}".format(
            dataset_dir, model.use_case,  model.framework, dataset_name, dataset_catalog, str(e)))

    # Train the model using the dataset
    try:
        model.train(dataset, output_dir=output_dir, epochs=epochs, initial_checkpoints=init_checkpoints)
    except Exception as e:
        sys.exit("There was an error during model training:\n{}".format(str(e)))

    # Save the trained model
    try:
        # TODO: also export info about how the model was trained, logs, etc so that it's reproducible
        model.export(output_dir)
    except Exception as e:
        sys.exit("There was an error when saving the model:\n{}".format(str(e)))
