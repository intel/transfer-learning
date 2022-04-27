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

from tlk.models.image_classification.tfhub_image_classification_model import tfhub_model_map
from tlk.utils.types import FrameworkType, UseCaseType

@click.group("list")
def list_group():
    """ Lists the available frameworks, use cases, or models """
    pass


@list_group.command("use_cases", help="List the available use cases")
def list_use_cases():
    print("\n".join([e.name.lower() for e in UseCaseType]))


@list_group.command("frameworks", help="List the available frameworks")
def list_frameworks():
    print("\n".join([e.name.lower() for e in FrameworkType]))


@list_group.command("models", help="List the available models")
@click.option("--framework", "-f",
              required=False,
              help="Filter the list of models by framework.")
def list_models(framework):
    # TODO: Standardize the way we store model info and get the list of models
    if framework and framework.lower() == "pytorch":
        print("There are no supported PyTorch models")

    print("Image Classification")
    print("-" * 40)
    image_classification_models = ["{} (tensorflow)".format(m) for m in tfhub_model_map.keys()]
    print("\n".join(image_classification_models))
    print("")
