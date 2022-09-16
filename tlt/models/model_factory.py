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
from pydoc import locate

from tlt import TLT_BASE_DIR
from tlt.utils.file_utils import read_json_file
from tlt.utils.types import FrameworkType, UseCaseType


model_map = {
    FrameworkType.TENSORFLOW: {
        UseCaseType.IMAGE_CLASSIFICATION: {
            "TFHub": {"module": "tlt.models.image_classification.tfhub_image_classification_model",
                     "class": "TFHubImageClassificationModel"},
            "Custom": {"module": "tlt.models.image_classification.tf_image_classification_model",
                      "class": "TFImageClassificationModel"}
        },
        UseCaseType.TEXT_CLASSIFICATION: {
            "TFHub": {
                "module": "tlt.models.text_classification.tfhub_text_classification_model",
                "class": "TFHubTextClassificationModel"},
            "Custom": {"module": "tlt.models.text_classification.tf_text_classification_model",
                      "class": "TFTextClassificationModel"}
        }
    },
    FrameworkType.PYTORCH: {
        UseCaseType.IMAGE_CLASSIFICATION: {
            "torchvision": {"module": "tlt.models.image_classification.torchvision_image_classification_model",
                            "class": "TorchvisionImageClassificationModel"},
            "Custom": {"module": "tlt.models.image_classification.pytorch_image_classification_model",
                      "class": "PyTorchImageClassificationModel"}
            }
        }
    }


def load_model(model_name: str, model, framework: FrameworkType = None, use_case: UseCaseType = None):
    """A factory method for loading an existing model.

        Args:
            model_name (str): name of model
            model (model or str): model object or directory with a saved_model.pb or model.pt file to load
            framework (str or FrameworkType): framework
            use_case (str or UseCaseType): use case

        Returns:
            model object

        Examples:
            >>> from tensorflow.keras import Sequential, Input
            >>> from tensorflow.keras.layers import Dense
            >>> from tlt.models.model_factory import load_model
            >>> my_model = Sequential([Input(shape=(3,)), Dense(4, activation='relu'), Dense(5, activation='softmax')])
            >>> model = load_model('my_model', my_model, 'tensorflow', 'image_classification')

    """

    if not isinstance(framework, FrameworkType):
        framework = FrameworkType.from_str(framework)

    if use_case is not None and not isinstance(use_case, UseCaseType):
        use_case = UseCaseType.from_str(use_case)
    
    model_class = locate('{}.{}'.format(model_map[framework][use_case]['Custom']['module'],
                                        model_map[framework][use_case]['Custom']['class']))
    return model_class(model_name, model)


def get_model(model_name: str, framework: FrameworkType = None):
    """A factory method for creating models.

        Args:
            model_name (str): name of model
            framework (str or FrameworkType): framework

        Returns:
            model object

        Raises:
            NotImplementedError if the model requested is not supported yet

        Example:
            >>> from tlt.models.model_factory import get_model
            >>> model = get_model('efficientnet_b0', 'tensorflow')
            >>> model.image_size
            224

    """

    if not isinstance(framework, FrameworkType):
        framework = FrameworkType.from_str(framework)

    model_use_case, model_dict = get_model_info(model_name, framework)

    if not model_use_case:
        framework_str = "tensorflow or pytorch" if framework is None else str(framework)
        raise ValueError("The specified model is not supported for {}: {}".format(framework_str, model_name))

    if len(model_dict) > 1:
        raise ValueError("Multiple frameworks support {}. Please specify a framework type.".format(model_name))

    model_framework_str = list(model_dict.keys())[0]
    model_framework_enum = FrameworkType.from_str(model_framework_str)
    model_hub = model_dict[model_framework_str]["model_hub"]

    if model_framework_enum in model_map:
        if model_use_case in model_map[model_framework_enum]:
            if model_hub in model_map[model_framework_enum][model_use_case]:
                model_class = locate('{}.{}'.format(model_map[model_framework_enum][model_use_case][model_hub]['module'],
                                                    model_map[model_framework_enum][model_use_case][model_hub]['class']))
                return model_class(model_name)

    raise NotImplementedError("Not implemented yet: {} {}".format(model_framework_str, model_name))


def get_supported_models(framework: FrameworkType = None, use_case: UseCaseType = None):
    """
    Returns a dictionary of supported models organized by use case, model name, and framework.
    The leaf items in the dictionary are attributes about the pretrained model.

    Args:
        framework (str or FrameworkType): framework
        use_case (str or UseCaseType): use case

    Returns:
        dictionary

    Raises:
        NameError if a model config file is found with an unknown or missing use case

    """
    # Directory of json files for the supported models
    config_directory = os.path.join(TLT_BASE_DIR, "models/configs")

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
    for config_filename in [x for x in os.listdir(config_directory) if os.path.isfile(os.path.join(config_directory, x))]:
        # Figure out which framework this config is, and filter it out, if necessary
        config_framework = FrameworkType.PYTORCH if 'torch' in config_filename else FrameworkType.TENSORFLOW

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

    Args:
        framework (str or FrameworkType): framework
        use_case (str or UseCaseType): use case
        verbose (boolean): include all model data from the config file in result, default is False

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

