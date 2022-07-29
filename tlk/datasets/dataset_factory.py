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

from pydoc import locate

from tlk.utils.types import FrameworkType, UseCaseType

dataset_map = {
    FrameworkType.TENSORFLOW: {
        UseCaseType.IMAGE_CLASSIFICATION: {
            "tf_datasets": {"module": "tlk.datasets.image_classification.tf_image_classification_dataset",
                            "class": "TFImageClassificationDataset"},
            "custom": {"module": "tlk.datasets.image_classification.tf_custom_image_classification_dataset",
                            "class": "TFCustomImageClassificationDataset"}
        },
        UseCaseType.TEXT_CLASSIFICATION: {
            "tf_datasets": {"module": "tlk.datasets.text_classification.tfds_text_classification_dataset",
                            "class": "TFDSTextClassificationDataset"}
        }
    },
    FrameworkType.PYTORCH: {
        UseCaseType.IMAGE_CLASSIFICATION: {
            "torchvision": {"module": "tlk.datasets.image_classification.torchvision_image_classification_dataset",
                            "class": "TorchvisionImageClassificationDataset"},
            "custom": {"module": "tlk.datasets.image_classification.pytorch_custom_image_classification_dataset",
                            "class": "PyTorchCustomImageClassificationDataset"}
        }
    }
}


def load_dataset(dataset_dir: str, use_case: UseCaseType, framework: FrameworkType, dataset_name=None, **kwargs):
    """A factory method for loading a custom dataset. Note that the directory of images is expected to be organized
    with subfolders for each image class. Each subfolder should contain .jpg images for the class. The name of the
    subfolder will be used as the class label.

    .. code-block:: text

        dataset_dir
          ├── class_a
          ├── class_b
          └── class_c

    Args:
        dataset_dir (str): directory containing the dataset
        use_case (str or UseCaseType): use case or task the dataset will be used to model
        framework (str or FrameworkType): framework
        dataset_name (str): optional; name of the dataset used for informational purposes
        **kwargs: optional; additional keyword arguments depending on the type of dataset being loaded

    Returns:
        (dataset)

    Raises:
        NotImplementedError if the type of dataset being loaded is not supported

    Example:
        >>> from tlk.datasets.dataset_factory import load_dataset
        >>> data = load_dataset('/tf_dataset/dataset/transfer_learning/flower_photos', 'image_classification', 'tensorflow')
        Found 3670 files belonging to 5 classes.
        >>> data.class_names
        ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    """
    if not isinstance(framework, FrameworkType):
        framework = FrameworkType.from_str(framework)

    if not isinstance(use_case, UseCaseType):
        use_case = UseCaseType.from_str(use_case)

    dataset_catalog = "custom"

    if framework in dataset_map.keys():
        if use_case in dataset_map[framework].keys():
            if dataset_catalog in dataset_map[framework][use_case]:
                dataset_class = locate('{}.{}'.format(dataset_map[framework][use_case][dataset_catalog]['module'],
                                                      dataset_map[framework][use_case][dataset_catalog]['class']))
                return dataset_class(dataset_dir, dataset_name, **kwargs)

    # If no match was found in the map, then it's not implemented yet
    raise NotImplementedError("Custom dataset support for {} {} {} has not been implemented yet".format(
        str(framework), str(use_case), dataset_catalog))


def get_dataset(dataset_dir: str, use_case: UseCaseType, framework: FrameworkType,
                dataset_name=None, dataset_catalog=None, **kwargs):
    """
    A factory method for using a dataset from a catalog.

    Args:
        dataset_dir (str): directory containing the dataset or to which the dataset should be downloaded
        use_case (str or UseCaseType): use case or task the dataset will be used to model
        framework (str or FrameworkType): framework
        dataset_name (str): optional; name of the dataset
        dataset_catalog (str): optional; catalog from which to download the dataset. If a dataset name is
                               provided and no dataset catalog is given, it will default to use tf_datasets
                               for a TensorFlow model, torchvision for PyTorch CV models, and huggingface
                               datasets for HuggingFace models.
        **kwargs: optional; additional keyword arguments for the framework or dataset_catalog

    Returns:
        (dataset)

    Raises:
        NotImplementedError if the dataset requested is not supported yet

    Example:
        >>> from tlk.datasets.dataset_factory import get_dataset
        >>> data = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers', 'tf_datasets')
        >>> data.class_names
        ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

    """
    if not isinstance(framework, FrameworkType):
        framework = FrameworkType.from_str(framework)

    if not isinstance(use_case, UseCaseType):
        use_case = UseCaseType.from_str(use_case)

    if dataset_name and not dataset_catalog:
        # Try to assume a dataset catalog based on the other information that we have
        if framework is FrameworkType.TENSORFLOW:
            dataset_catalog = "tf_datasets"
        elif framework is FrameworkType.PYTORCH and \
            use_case in [UseCaseType.IMAGE_CLASSIFICATION, UseCaseType.OBJECT_DETECTION]:
            dataset_catalog = "torchvision"
        else:
            dataset_catalog = "huggingface"

        print("Using dataset catalog '{}', since no dataset catalog was specified".format(dataset_catalog))

    if framework in dataset_map.keys():
        if use_case in dataset_map[framework].keys():
            if dataset_catalog and dataset_catalog in dataset_map[framework][use_case]:
                dataset_class = locate('{}.{}'.format(dataset_map[framework][use_case][dataset_catalog]['module'],
                                                      dataset_map[framework][use_case][dataset_catalog]['class']))
                return dataset_class(dataset_dir, dataset_name, **kwargs)

    # If no match was found in the map, then it's not implemented yet
    raise NotImplementedError("Datasets support for {} {} {} has not been implemented yet".format(
        str(framework), str(use_case), dataset_catalog))
