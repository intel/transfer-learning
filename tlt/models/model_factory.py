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

import os
from pydoc import locate

from tlt import TLT_BASE_DIR
from tlt.utils.file_utils import read_json_file
from tlt.utils.types import FrameworkType, UseCaseType


model_map = {
    FrameworkType.TENSORFLOW:
    {
        UseCaseType.IMAGE_CLASSIFICATION:
        {
            "TFHub": {"module": "tlt.models.image_classification.tfhub_image_classification_model",
                      "class": "TFHubImageClassificationModel"},
            "Keras": {"module": "tlt.models.image_classification.keras_image_classification_model",
                      "class": "KerasImageClassificationModel"},
            "Custom": {"module": "tlt.models.image_classification.tf_image_classification_model",
                       "class": "TFImageClassificationModel"}
        },
        UseCaseType.TEXT_CLASSIFICATION:
        {
            "huggingface": {"module": "tlt.models.text_classification.tf_hf_text_classification_model",
                            "class": "TFHFTextClassificationModel"},
            "TFHub": {"module": "tlt.models.text_classification.tfhub_text_classification_model",
                      "class": "TFHubTextClassificationModel"},

            "Custom": {"module": "tlt.models.text_classification.tf_text_classification_model",
                       "class": "TFTextClassificationModel"}
        }
    },
    FrameworkType.PYTORCH:
    {
        UseCaseType.IMAGE_CLASSIFICATION:
        {
            "torchvision": {"module": "tlt.models.image_classification.torchvision_image_classification_model",
                            "class": "TorchvisionImageClassificationModel"},
            "pytorch_hub": {"module": "tlt.models.image_classification.pytorch_hub_image_classification_model",
                            "class": "PyTorchHubImageClassificationModel"},
            "Custom": {"module": "tlt.models.image_classification.pytorch_image_classification_model",
                       "class": "PyTorchImageClassificationModel"}
        },
        UseCaseType.TEXT_CLASSIFICATION: {
            "huggingface": {"module": "tlt.models.text_classification.pytorch_hf_text_classification_model",
                            "class": "PyTorchHFTextClassificationModel"},
        },
        UseCaseType.IMAGE_ANOMALY_DETECTION:
        {
            "torchvision": {"module": "tlt.models.image_anomaly_detection.torchvision_image_anomaly_detection_model",
                            "class": "TorchvisionImageAnomalyDetectionModel"},
            "Custom": {"module": "tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model",
                       "class": "PyTorchImageAnomalyDetectionModel"}
        }
    }
}


def load_model(model_name: str, model, framework: FrameworkType = None, use_case: UseCaseType = None,
               model_hub: str = None, **kwargs):
    """A factory method for loading an existing model.

        Args:
            model_name (str): name of model
            model (model or str): model object or directory with a saved_model.pb or model.pt file to load
            framework (str or FrameworkType): framework
            use_case (str or UseCaseType): use case
            model_hub (str): The model hub where the model originated
            kwargs: optional; additional keyword arguments for optimizer and loss function configuration.
                The `optimizer` and `loss` arguments can be set to Optimizer and Loss classes, depending on the model's
                framework (examples: `optimizer=tf.keras.optimizers.Adam` for TensorFlow,
                `loss=torch.nn.CrossEntropyLoss` for PyTorch). Additional keywords for those classes' initialization
                can then be provided to further configure the objects when they are created (example: `amsgrad=True`
                for the PyTorch Adam optimizer). Refer to the framework documentation for the function you want to use.

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

    model_hub = model_hub if model_hub else 'Custom'

    model_class = locate('{}.{}'.format(model_map[framework][use_case][model_hub]['module'],
                                        model_map[framework][use_case][model_hub]['class']))
    return model_class(model_name, model, **kwargs)


def get_model(model_name: str, framework: FrameworkType = None, use_case: UseCaseType = None, **kwargs):
    """A factory method for creating models.

        Args:
            model_name (str): name of model
            framework (str or FrameworkType): framework
            use_case (str or FrameworkType): use case
            kwargs: optional; additional keyword arguments for optimizer and loss function configuration.
                The `optimizer` and `loss` arguments can be set to Optimizer and Loss classes, depending on the model's
                framework (examples: `optimizer=tf.keras.optimizers.Adam` for TensorFlow,
                `loss=torch.nn.CrossEntropyLoss` for PyTorch). Additional keywords for those classes' initialization
                can then be provided to further configure the objects when they are created (example: `amsgrad=True`
                for the PyTorch Adam optimizer). Refer to the framework documentation for the function you want to use.

        Returns:
            model object

        Raises:
            NotImplementedError: if the model requested is not supported yet

        Example:
            >>> from tlt.models.model_factory import get_model
            >>> model = get_model('efficientnet_b0', 'tensorflow')
            >>> model.image_size
            224

    """

    if not isinstance(framework, FrameworkType):
        framework = FrameworkType.from_str(framework)

    if use_case is not None and not isinstance(use_case, UseCaseType):
        use_case = UseCaseType.from_str(use_case)

    model_info = get_model_info(model_name, framework, use_case)
    valid_use_cases = list(model_info.keys())

    if not valid_use_cases or (use_case is not None and use_case not in valid_use_cases):
        framework_str = "tensorflow or pytorch" if framework is None else str(framework)
        raise ValueError("The specified model is not supported for {}: {}".format(framework_str, model_name))
    elif len(valid_use_cases) == 1:
        model_use_case = valid_use_cases[0]
        model_dict = model_info[model_use_case]
    else:
        # Default to image classification for backward compatibility
        if UseCaseType.IMAGE_CLASSIFICATION in valid_use_cases:
            model_use_case = UseCaseType.IMAGE_CLASSIFICATION
            model_dict = model_info[UseCaseType.IMAGE_CLASSIFICATION]
        else:
            raise ValueError("More than one use case applies for {}. Please specify: {}".format(model_name,
                                                                                                valid_use_cases))
    if not model_use_case:
        framework_str = "tensorflow or pytorch" if framework is None else str(framework)
        raise ValueError("The specified model is not supported for {}: {}".format(framework_str, model_name))

    if len(model_dict) > 1:
        raise ValueError("Multiple frameworks support {}. Please specify a framework type.".format(model_name))

    model_framework_str = list(model_dict.keys())[0]
    model_fw_enum = FrameworkType.from_str(model_framework_str)
    model_hub = model_dict[model_framework_str]["model_hub"]

    if model_fw_enum in model_map:
        if model_use_case in model_map[model_fw_enum]:
            if model_hub in model_map[model_fw_enum][model_use_case]:
                model_class = locate('{}.{}'.format(model_map[model_fw_enum][model_use_case][model_hub]['module'],
                                                    model_map[model_fw_enum][model_use_case][model_hub]['class']))
                return model_class(model_name, **kwargs)

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
        NameError: if a model config file is found with an unknown or missing use case

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
    for config_file in [x for x in os.listdir(config_directory) if os.path.isfile(os.path.join(config_directory, x))]:
        # Figure out which framework this config is, and filter it out, if necessary
        config_framework = FrameworkType.TENSORFLOW if 'tf' in config_file else FrameworkType.PYTORCH

        if framework is not None and framework != config_framework:
            continue

        # Figure out which use case this config file is for
        config_use_case = None
        for uc in UseCaseType:
            if str(uc) in config_file:
                config_use_case = str(uc)
                break

        if config_use_case is None:
            raise NameError("The config file {} does not match any of the supported use case types".format(
                config_file))

        # Filter the config file out, by use case, if necessary
        if use_case is not None and str(use_case) != config_use_case:
            continue

        # If it hasn't been filtered out, then read the config from the json file
        config_dict = read_json_file(os.path.join(config_directory, config_file))

        for model_name in config_dict.keys():
            if model_name not in models[str(config_use_case)].keys():
                models[str(config_use_case)][model_name] = {}

            models[str(config_use_case)][model_name][str(config_framework)] = config_dict[model_name]

    return models


def print_supported_models(framework: FrameworkType = None, use_case: UseCaseType = None, verbose: bool = False,
                           markdown: bool = False):
    """
    Prints a list of the supported models, categorized by use case. The results can be filtered to only show a given
    framework or use case.

    Args:
        framework (str or FrameworkType): framework
        use_case (str or UseCaseType): use case
        verbose (boolean): include all model data from the config file in result, default is False
        markdown (boolean): Print results as markdown tables (used for updating documentation).
                            Not compatible with verbose=True.

    """
    models = get_supported_models(framework, use_case)

    # Proper names
    model_hub_map = {
        "torchvision": "Torchvision",
        "tfhub": "TensorFlow Hub",
        "pytorch_hub": "PyTorch Hub",
        "huggingface": "Hugging Face",
        "keras": "Keras Applications"
    }
    framework_name_map = {
        "tensorflow": "TensorFlow",
        "pytorch": "PyTorch"
    }

    for model_use_case in models.keys():
        if markdown:
            print("## {}\n".format(model_use_case.replace("_", " ").title()))
        else:
            print("-" * 30)
            print(model_use_case.replace("_", " ").upper())
            print("-" * 30)

        if len(models[model_use_case].keys()) == 0:
            filter_message = ""
            if framework is not None:
                filter_message = "for {}".format(str(framework))
            print("No {} models are supported at this time {}\n".format(
                model_use_case.replace("_", " "), filter_message))
            continue

        if markdown:
            print("| Model name | Framework | Model Hub |")
            print("|------------|-----------|-----------|")

        # Get a sorted list of model names
        model_name_list = list(models[model_use_case].keys())
        model_name_list.sort(key=str.swapcase)

        for model_name in model_name_list:
            for model_framework in models[model_use_case][model_name].keys():
                model_hub = models[model_use_case][model_name][model_framework]["model_hub"] if \
                    "model_hub" in models[model_use_case][model_name][model_framework].keys() else ""
                model_hub_display = model_hub_map[model_hub.lower()] if model_hub.lower() in model_hub_map.keys() \
                    else model_hub
                model_framework_display = framework_name_map[model_framework.lower()] if \
                    model_framework.lower() in framework_name_map.keys() else model_framework

                if markdown:
                    print("| {} | {} | {} |".format(model_name, model_framework_display, model_hub_display))
                else:
                    print("{} ({} model from {})".format(model_name, model_framework_display, model_hub_display))

                if verbose and not markdown:
                    for model_attribute, attribute_value in models[model_use_case][model_name][model_framework].items():
                        print("    {}: {}".format(model_attribute, attribute_value))

        # Empty line between use cases
        print("")


def get_model_info(model_name, framework=None, use_case=None):
    models = get_supported_models(framework, use_case)
    info = {}

    for model_use_case in models.keys():
        if model_name in models[model_use_case]:
            # Found a matching model
            info[UseCaseType.from_str(model_use_case)] = models[model_use_case][model_name]

    return info
