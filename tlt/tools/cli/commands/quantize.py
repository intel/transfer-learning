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
import datetime
import os
import sys

from tlt.utils.types import FrameworkType


@click.command()
@click.option("--model-dir", "--model_dir",
              required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Model directory to reload for quantization. The model directory should contain a saved_model.pb "
                   "for TensorFlow models or a model.pt file for PyTorch models.")
@click.option("--dataset-dir", "--dataset_dir",
              required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Dataset directory for a custom dataset. Quantization is not supported with dataset catalogs at "
                   "this time.")
@click.option("--inc-config", "--inc_config",
              required=False,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to a config file (yaml) that will be used to quantize the model using the Intel Neural "
                   "Compressor. The INC config examples for quantization can be found at: "
                   "https://github.com/intel/neural-compressor/tree/master/examples. "
                   "If no INC config file is provided, a default config file will be generated.")
@click.option("--batch-size", "--batch_size",
              required=False,
              type=click.IntRange(min=1),
              default=32,
              show_default=True,
              help="Batch size used during quantization, if an INC config file is not provided. If an INC config file "
                   "is provided, the batch size from the config file will be used.")
@click.option("--accuracy-criterion", "--accuracy_criterion",
              required=False,
              type=click.FloatRange(min=0, max=1.0),
              default=0.01,
              show_default=True,
              help="Relative accuracy loss to allow (for example, a value of 0.01 allows for a relative accuracy "
                   "loss of 1%), if an INC config file is not provided. If an INC config file is provided, the "
                   "accuracy criterion from the config file will be used.")
@click.option("--timeout",
              required=False,
              type=click.IntRange(min=0),
              default=0,
              show_default=True,
              help="Tuning timeout in seconds, if an INC config file is not provided. If an INC config file is "
                   "provided, the timeout from the config file will be used. Tuning processing finishes when the "
                   "timeout or max trials is reached. A tuning timeout of 0 means that the tuning phase stops when "
                   "the accuracy criterion is met.")
@click.option("--max-trials", "--max_trials",
              required=False,
              type=click.IntRange(min=0),
              default=50,
              show_default=True,
              help="Maximum number of tuning trials, if an INC config file is not provided. If an INC config file is "
                   "provided, the number of max trials from the config file will be used. Tuning processing finishes "
                   "when the timeout or max trials is reached.")
@click.option("--output-dir", "--output_dir",
              required=True,
              type=click.Path(file_okay=False),
              help="A writeable output directory. The output directory will be used as a location to save the "
                   "quantized model, the tuning workspace, and the INC config file, if a config file is not provided.")
def quantize(model_dir, dataset_dir, inc_config, batch_size, accuracy_criterion, timeout, max_trials, output_dir):
    """
    Uses the Intel Neural Compressor to perform post-training quantization on a trained model
    """
    print("Model directory:", model_dir)
    print("Dataset directory:", dataset_dir)

    if inc_config:
        print("INC config file:", inc_config)
    else:
        print("Accuracy criterion:", accuracy_criterion)
        print("Exit policy timeout:", timeout)
        print("Exit policy max trials:", max_trials)
        print("Batch size:", batch_size)

    print("Output directory:", output_dir)

    try:
        # Create the output directory, if it doesn't exist
        from tlt.utils.file_utils import verify_directory
        verify_directory(output_dir, require_directory_exists=False)
    except Exception as e:
        sys.exit("Error while verifying the output directory: {}", str(e))

    saved_model_path = os.path.join(model_dir, "saved_model.pb")
    pytorch_model_path = os.path.join(model_dir, "model.pt")
    if os.path.isfile(saved_model_path):
        framework = FrameworkType.TENSORFLOW
    elif os.path.isfile(pytorch_model_path):
        framework = FrameworkType.PYTORCH
    else:
        sys.exit("Quantization is currently only implemented for TensorFlow saved_model.pb and PyTorch model.pt "
                 "models. No such files found in the model directory ({}).".format(model_dir))

    # Get the model name from the directory path, assuming models are exported like <model name>/n
    model_name = os.path.basename(os.path.dirname(model_dir))

    print("Model name:", model_name)
    print("Framework:", framework)

    try:
        from tlt.models.model_factory import get_model

        model = get_model(model_name, framework)
    except Exception as e:
        sys.exit("An error occurred while getting the model: {}\nNote that the model directory is expected to contain "
                 "a previously exported model where the directory structure is <model name>/n/saved_model.pb "
                 "(for TensorFlow) or <model name>/n/model.pt (for PyTorch).".format(str(e)))

    try:

        from tlt.datasets import dataset_factory

        dataset = dataset_factory.load_dataset(dataset_dir, model.use_case, model.framework)

        # Generate a default inc config file, if one was not provided by the user
        if not inc_config:
            now = datetime.datetime.now()
            dt_str = now.strftime("%y%m%d%H%M%S")
            inc_config = os.path.join(output_dir, "{}_config_{}.yaml".format(model_name, dt_str))
            model.write_inc_config_file(inc_config, dataset, batch_size=batch_size, overwrite=True,
                                        exit_policy_timeout=timeout, exit_policy_max_trials=max_trials,
                                        accuracy_criterion_relative=accuracy_criterion,
                                        tuning_workspace=os.path.join(output_dir, "nc_workspace"))

        # Setup a directory for the quantized model
        quantized_output_dir = os.path.join(output_dir, "quantized", model_name)
        verify_directory(quantized_output_dir)
        if len(os.listdir(quantized_output_dir)) > 0:
            quantized_output_dir = os.path.join(quantized_output_dir, "{}".format(
                len(os.listdir(quantized_output_dir)) + 1))
        else:
            quantized_output_dir = os.path.join(quantized_output_dir, "1")

        # Call the quantization API
        print("Starting post-training quantization", flush=True)
        model.quantize(model_dir, quantized_output_dir, inc_config)

    except Exception as e:
        sys.exit("An error occurred during quantization: {}".format(str(e)))
