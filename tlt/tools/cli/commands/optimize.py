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
import os
import sys

from tlt.utils.types import FrameworkType


@click.command()
@click.option("--model-dir", "--model_dir",
              required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Model directory to reload for graph optimization. The model directory should contain a "
                   "saved_model.pb TensorFlow model.")
@click.option("--output-dir", "--output_dir",
              required=True,
              type=click.Path(file_okay=False),
              help="A writeable output directory. The output directory will be used as a location to save the "
                   "optimized model.")
def optimize(model_dir, output_dir):
    """
    Uses the Intel Neural Compressor to perform graph optimization on a trained model
    """
    print("Model directory:", model_dir)
    print("Output directory:", output_dir)

    try:
        # Create the output directory, if it doesn't exist
        from tlt.utils.file_utils import verify_directory
        verify_directory(output_dir, require_directory_exists=False)
    except Exception as e:
        sys.exit("Error while verifying the output directory: {}", str(e))

    saved_model_path = os.path.join(model_dir, "saved_model.pb")
    # pytorch_model_path = os.path.join(model_dir, "model.pt")
    if os.path.isfile(saved_model_path):
        framework = FrameworkType.TENSORFLOW
    else:
        sys.exit("Graph optimization is currently only supported for TensorFlow saved_model.pb "
                 "models. No such files found in the model directory ({}).".format(model_dir))

    # Get the model name from the directory path, assuming models are exported like <model name>/n
    model_name = os.path.basename(os.path.dirname(model_dir))

    print("Model name:", model_name)
    print("Framework:", framework)

    try:
        from tlt.models.model_factory import get_model

        model = get_model(model_name, framework)
        model.load_from_directory(model_dir)
    except Exception as e:
        sys.exit("An error occurred while getting the model: {}\nNote that the model directory is expected to contain "
                 "a previously exported model where the directory structure is <model name>/n/saved_model.pb "
                 "(for TensorFlow).".format(str(e)))

    try:
        # Setup a directory for the quantized model
        optimized_output_dir = os.path.join(output_dir, "optimized", model_name)
        verify_directory(optimized_output_dir)
        if len(os.listdir(optimized_output_dir)) > 0:
            optimized_output_dir = os.path.join(optimized_output_dir, "{}".format(
                len(os.listdir(optimized_output_dir)) + 1))
        else:
            optimized_output_dir = os.path.join(optimized_output_dir, "1")

        # Call the graph optimization API
        print("Starting graph optimization", flush=True)
        model.optimize_graph(optimized_output_dir)

    except Exception as e:
        sys.exit("An error occurred during graph optimization: {}".format(str(e)))
