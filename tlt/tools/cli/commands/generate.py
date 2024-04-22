#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

from tlt.utils.file_utils import get_model_name_from_path


@click.command()
@click.option("--model-name", "--model_name",
              required=False,
              type=str,
              help="Name of the model to use")
@click.option("--model-dir", "--model_dir",
              required=False,
              type=str,
              help="Model directory to reload a previously exported model.")
@click.option("--prompt",
              required=False,
              type=str,
              help="Prompt with added context used to build the prompt dictionary")
@click.option("--temperature",
              required=False,
              type=float,
              default=1.0,
              help="The value used to modulate the next token probabilities (default: 1.0)")
@click.option("--top-p", "--top_p",
              required=False,
              type=float,
              default=0.75,
              help="If set to float < 1, only the smallest set of most probable tokens with probabilities "
                   "that add up to top_p or higher are kept for generation (default: 0.75)")
@click.option("--top-k", "--top_k",
              required=False,
              type=int,
              default=40,
              help="The number of highest probability vocabulary tokens to keep for top-k-filtering (default: 40)")
@click.option("--repetition-penalty", "--repetition_penalty",
              required=False,
              type=float,
              default=1.0,
              help="The parameter for repetition penalty. 1.0 means no penalty. (default: 1.0)")
@click.option("--num-beams", "--num_beams",
              required=False,
              type=int,
              default=4,
              help="Number of beams for beam search. 1 means no beam search. (default: 4)")
@click.option("--max-new-tokens", "--max_new_tokens",
              required=False,
              type=int,
              default=128,
              help="The maximum number of new tokens generated (default: 128)")
def generate(model_dir, model_name, prompt, temperature, top_p, top_k, repetition_penalty,
             num_beams, max_new_tokens):
    """
    Generates text from the model
    """
    from tlt.models import model_factory
    if model_name is None and model_dir is None:
        sys.exit("ERROR: Please define a model_dir to load a saved model OR specify model_name to use a "
                 "model for text generation")
    if model_name is None:
        model_name = get_model_name_from_path(model_dir)

    if model_dir:
        if not os.path.exists(os.path.join(model_dir, 'training_args.bin')):
            sys.exit("The Generate command is only supported for Pytorch Text Generation models")

    # Get the model
    try:
        model = model_factory.get_model(model_name, framework="pytorch", use_case="text_generation")
    except Exception as e:
        sys.exit("Error while getting the model (model name: {}), The Generate command is only supported"
                 " for Pytorch Text Generation models:\n{}".format(model_name, str(e)))

    if model_dir:
        if os.path.exists(model_dir):
            model.load_from_directory(model_dir)

    prompt = prompt.replace("\\n", "\n")
    output = model.generate(prompt, temperature=temperature, repetition_penalty=repetition_penalty, top_p=top_p,
                            top_k=top_k, num_beams=num_beams, max_new_tokens=max_new_tokens)
    print(*output, sep='\n')
