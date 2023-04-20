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
import sys

from tlt.utils.types import FrameworkType, UseCaseType


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
@click.option("--use-case", "--use_case",
              required=False,
              help="Filter the list of models to a single use case")
@click.option("--verbose", "verbose",
              flag_value=True,
              default=False,
              help="Verbose output with extra information about each model")
@click.option("--markdown",
              flag_value=True,
              default=False,
              hidden=True,
              help="Display the results as markdown. Not compatible with --verbose.")
def list_models(framework, use_case, verbose, markdown):
    """
    List the supported models and the information that we have about each model from the config files.
    """
    from tlt.models.model_factory import print_supported_models

    try:
        print_supported_models(framework, use_case, verbose, markdown)
    except Exception as e:
        sys.exit("Error while listing the supported models for framework: {}, use case: {}\n  {}".format(
            str(framework), str(use_case), str(e)))
