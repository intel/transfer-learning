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
import shutil
import sys
import tempfile

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
@click.option("--inc-config", "--inc_config",
              required=False,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to a config file (yaml) that will be used to benchmark the model using the Intel Neural "
                   "Compressor. The INC benchmarking documentation can be found at: "
                   "https://github.com/intel/neural-compressor/blob/master/docs/benchmark.md "
                   "If no INC config file is provided, a default config file will be generated.")
@click.option("--mode",
              required=False,
              type=click.Choice(['performance', 'accuracy'], case_sensitive=False),
              default='performance',
              show_default=True,
              help="Specify to benchmark the model's performance or accuracy")
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
def benchmark(model_dir, dataset_dir, inc_config, mode, batch_size, output_dir):
    """
    Uses the Intel Neural Compressor to benchmark a trained model
    """
    print("Model directory:", model_dir)
    print("Dataset directory:", dataset_dir)
    print("Benchmarking mode:", mode)

    if inc_config:
        print("INC config file:", inc_config)
    else:
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
        dataset = dataset_factory.load_dataset(dataset_dir, model.use_case, model.framework)

        # Generate a default inc config file, if one was not provided by the user
        if not inc_config:
            if not output_dir:
                output_dir = tempfile.mkdtemp()
                temp_dir = output_dir
            now = datetime.datetime.now()
            dt_str = now.strftime("%y%m%d%H%M%S")
            inc_config = os.path.join(output_dir, "{}_config_{}.yaml".format(model_name, dt_str))
            print("Writing INC config file to {}".format(inc_config))
            model.write_inc_config_file(inc_config, dataset, batch_size, overwrite=True)

        # Call the benchmarking API
        print("Starting benchmarking", flush=True)
        model.benchmark(model_dir, inc_config, mode)

    except Exception as e:
        sys.exit("An error occurred during benchmarking: {}".format(str(e)))
    finally:
        # Remove the temp directory, if we created one
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
