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

import os

from tlk import TLK_BASE_DIR
from tlk.utils.file_utils import read_json_file
from tlk.utils.types import FrameworkType, UseCaseType


def get_model_class_map():
    from tlk.models.image_classification.tfhub_image_classification_model import TFHubImageClassificationModel
    return {
        FrameworkType.TENSORFLOW: {
            UseCaseType.IMAGE_CLASSIFICATION: {
                "TFHub": TFHubImageClassificationModel
            }
        }
    }


def get_model(model_name: str, framework: FrameworkType = None):
    """A factory method for creating models.

        Args:
            model_name (str): name of model
            framework (str or FrameworkType): framework

        Returns:
            model object

        Raises:
            NotImplementedError if the model requested is not supported yet
    """

    if not isinstance(framework, FrameworkType):
        framework = FrameworkType.from_str(framework)

    if framework == FrameworkType.PYTORCH:
        raise NotImplementedError("PyTorch support has not been implemented")

    model_use_case, model_dict = get_model_info(model_name, framework)

    if not model_use_case:
        framework_str = "tensorflow or pytorch" if framework is None else str(framework)
        raise ValueError("The specified model is not supported for {}: {}".format(framework_str, model_name))

    if len(model_dict) > 1:
        raise ValueError("Multiple frameworks support {}. Please specify a framework type.".format(model_name))

    model_framework_str = list(model_dict.keys())[0]
    model_framework_enum = FrameworkType.from_str(model_framework_str)
    model_hub = model_dict[model_framework_str]["model_hub"]

    # Get the map of model to the class implementation
    model_class_map = get_model_class_map()

    if model_framework_enum in model_class_map:
        if model_use_case in model_class_map[model_framework_enum]:
            if model_hub in model_class_map[model_framework_enum][model_use_case]:
                model_class = model_class_map[model_framework_enum][model_use_case][model_hub]
                return model_class(model_name)

    raise NotImplementedError("Not implemented yet: {} {}".format(model_framework_str, model_name))


def get_supported_models(framework: FrameworkType = None, use_case: UseCaseType = None):
    """
    Returns a dictionary of supported models organized by use case, model name, and framework.
    The leaf items in the dictionary are attributes about the pretrained model.
    """
    # Directory of json files for the supported models
    config_directory = os.path.join(TLK_BASE_DIR, "models/configs")

    # Models dictionary with keys for use case / model name / framework / model info
    models = {}

    if framework is not None and not isinstance(framework, FrameworkType):
        framework = FrameworkType.from_str(framework)

    if use_case is not None and not isinstance(use_case, UseCaseType):
        use_case = UseCaseType.from_str(use_case)

    # Initialize the models dictionary by use case
    if use_case is None:
        for uc in UseCaseType:
            models[str(uc)] = {}
    else:
        models[str(use_case)] = {}

    # Read configs into the models dictionary
    for config_filename in os.listdir(config_directory):
        # Figure out which framework this config is, and filter it out, if necessary
        config_framework = FrameworkType.PYTORCH if config_filename.startswith("pytorch") else FrameworkType.TENSORFLOW

        if framework is not None and framework != config_framework:
            continue

        # Figure out which use case this config file is for
        config_use_case = None
        for uc in UseCaseType:
            if str(uc) in config_filename:
                config_use_case = str(uc)
                break

        if config_use_case is None:
            raise NameError("The config file {} does not match any of the supported use case types".format(
                config_filename))

        # Filter the config file out, by use case, if necessary
        if use_case is not None and str(use_case) != config_use_case:
            continue

        # If it hasn't been filtered out, then read the config from the json file
        config_dict = read_json_file(os.path.join(config_directory, config_filename))

        for model_name in config_dict.keys():
            if model_name not in models[str(config_use_case)].keys():
                models[str(config_use_case)][model_name] = {}

            models[str(config_use_case)][model_name][str(config_framework)] = config_dict[model_name]

    return models


def print_supported_models(framework: FrameworkType = None, use_case: UseCaseType = None, verbose: bool = False):
    """
    Prints a list of the supported models, categorized by use case. The results can be filtered to only show a given
    framework or use case.
    """
    models = get_supported_models(framework, use_case)

    for model_use_case in models.keys():
        print("-" * 30)
        print(model_use_case.replace("_", " ").upper())
        print("-" * 30)

        if len(models[model_use_case].keys()) == 0:
            filter_message = ""
            if framework is not None:
                filter_message = "for {}".format(str(framework))
            print("No {} models are supported at this time {}".format(model_use_case.replace("_", " "), filter_message))

        # Get a sorted list of model names
        model_name_list = list(models[model_use_case].keys())
        model_name_list.sort()

        for model_name in model_name_list:
            for model_framework in models[model_use_case][model_name].keys():

                print("{} ({})".format(model_name, model_framework))

                if verbose:
                    for model_attribute, attribute_value in models[model_use_case][model_name][model_framework].items():
                        print("    {}: {}".format(model_attribute, attribute_value))

        # Empty line between use cases
        print("")


def get_model_info(model_name, framework=None):
    models = get_supported_models(framework)

    for model_use_case in models.keys():
        if model_name in models[model_use_case]:
            # Found a matching model
            return UseCaseType.from_str(model_use_case), models[model_use_case][model_name]

    return None, {}

