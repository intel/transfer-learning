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
import shutil
import sys

from tlt.utils.types import FrameworkType


@click.command()
@click.option("--model-dir", "--model_dir",
              required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Model directory to reload for benchmarking. The model directory should contain a saved_model.pb for"
                   " TensorFlow models or a model.pt file for PyTorch models.")
@click.option("--dataset-dir", "--dataset_dir",
              required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Dataset directory for a custom dataset. Benchmarking is not supported with dataset catalogs at "
                   "this time.")
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
@click.option("--batch-size", "--batch_size",
              required=False,
              type=click.IntRange(min=1),
              default=32,
              show_default=True,
              help="Batch size used for benchmarking, if an INC config file is not provided. If an INC config file is "
                   "provided, the batch size from the config file will be used.")
@click.option("--output-dir", "--output_dir",
              required=False,
              type=click.Path(file_okay=False),
              help="A writeable output directory. The output directory will be used as a location to write the INC "
                   "config file, if a config file is not provided. If no output directory is provided, a temporary "
                   "folder will be created and then deleted after benchmarking has completed.")
def benchmark(model_dir, dataset_dir, batch_size, output_dir, dataset_file, delimiter):
    """
    Uses the Intel Neural Compressor to benchmark a trained model
    """
    print("Model directory:", model_dir)
    print("Dataset directory:", dataset_dir)
    print("Batch size:", batch_size)

    if output_dir:
        print("Output directory:", output_dir)

    saved_model_path = os.path.join(model_dir, "saved_model.pb")
    pytorch_model_path = os.path.join(model_dir, "model.pt")
    if os.path.isfile(saved_model_path):
        framework = FrameworkType.TENSORFLOW
    elif os.path.isfile(pytorch_model_path):
        framework = FrameworkType.PYTORCH
    else:
        sys.exit("Benchmarking is currently only implemented for TensorFlow saved_model.pb and PyTorch model.pt "
                 "models. No such files found in the model directory ({}).".format(model_dir))
    model_name = os.path.basename(os.path.dirname(model_dir))

    print("Model name:", model_name)
    print("Framework:", framework)
    temp_dir = None

    try:
        from tlt.models.model_factory import get_model

        model = get_model(model_name, framework)
    except Exception as e:
        sys.exit("An error occurred while getting the model: {}\nNote that the model directory is expected to contain "
                 "a previously exported model where the directory structure is <model name>/n/saved_model.pb "
                 "(for TensorFlow) or <model name>/n/model.pt (for PyTorch).".format(str(e)))

    try:
        from tlt.datasets import dataset_factory
        if str(model.use_case) == "image_classification":
            dataset = dataset_factory.load_dataset(dataset_dir, model.use_case, model.framework)
        elif str(model.use_case) == 'text_classification':
            if not dataset_file:
                raise ValueError("Loading a text classification dataset requires --dataset-file to specify the "
                                 "file name of the .csv file to load from the --dataset-dir.")
            if not delimiter:
                raise ValueError("Loading a text classification dataset requires --delimiter in order to read the "
                                 ".csv file from the --dataset-dir. in the correct format")

            dataset = dataset_factory.load_dataset(dataset_dir, model.use_case, model.framework,
                                                   csv_file_name=dataset_file, delimiter=delimiter)
        else:
            sys.exit("ERROR: Benchmarking is currently only implemented for Image Classification "
                     "and Text Classification models")

        # Preprocess, batch, and split
        if 'image_size' in inspect.getfullargspec(dataset.preprocess).args:  # For Image classification
            dataset.preprocess(image_size=model.image_size, batch_size=batch_size)
        elif 'model_name' in inspect.getfullargspec(dataset.preprocess).args:  # For HF Text classification
            dataset.preprocess(model_name=model_name, batch_size=batch_size)
        else:  # For TF Text classification
            dataset.preprocess(batch_size=batch_size)
        dataset.shuffle_split()

        # Call the benchmarking API
        print("Starting benchmarking", flush=True)
        try:
            model.benchmark(dataset, saved_model_dir=model_dir)
        except TypeError:
            model.load_from_directory(model_dir)
            model.benchmark(dataset)
        except AttributeError:
            model._model = model._get_hub_model(model_name, len(dataset.class_names))
            model.benchmark(dataset, saved_model_dir=model_dir)

    except Exception as e:
        sys.exit("An error occurred during benchmarking: {}".format(str(e)))
    finally:
        # Remove the temp directory, if we created one
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
