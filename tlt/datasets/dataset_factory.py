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

from pydoc import locate

from tlt.utils.types import FrameworkType, UseCaseType

dataset_map = {
    FrameworkType.TENSORFLOW: {
        UseCaseType.IMAGE_CLASSIFICATION: {
            "tf_datasets": {"module": "tlt.datasets.image_classification.tfds_image_classification_dataset",
                            "class": "TFDSImageClassificationDataset"},
            "custom": {"module": "tlt.datasets.image_classification.tf_custom_image_classification_dataset",
                       "class": "TFCustomImageClassificationDataset"}
        },
        UseCaseType.TEXT_CLASSIFICATION: {
            "tf_datasets": {"module": "tlt.datasets.text_classification.tfds_text_classification_dataset",
                            "class": "TFDSTextClassificationDataset"},
            "custom": {"module": "tlt.datasets.text_classification.tf_custom_text_classification_dataset",
                       "class": "TFCustomTextClassificationDataset"}
        }
    },
    FrameworkType.PYTORCH: {
        UseCaseType.IMAGE_CLASSIFICATION: {
            "torchvision": {"module": "tlt.datasets.image_classification.torchvision_image_classification_dataset",
                            "class": "TorchvisionImageClassificationDataset"},
            "custom": {"module": "tlt.datasets.image_classification.pytorch_custom_image_classification_dataset",
                       "class": "PyTorchCustomImageClassificationDataset"}
        },
        UseCaseType.TEXT_CLASSIFICATION: {
            "huggingface": {"module": "tlt.datasets.text_classification.hf_text_classification_dataset",
                            "class": "HFTextClassificationDataset"},
            "custom": {"module": "tlt.datasets.text_classification.hf_custom_text_classification_dataset",
                       "class": "HFCustomTextClassificationDataset"}
        },
        UseCaseType.IMAGE_ANOMALY_DETECTION: {
            "custom": {"module": "tlt.datasets.image_anomaly_detection.pytorch_custom_image_anomaly_detection_dataset",
                       "class": "PyTorchCustomImageAnomalyDetectionDataset"}
        },
    }
}


def load_dataset(dataset_dir: str, use_case: UseCaseType, framework: FrameworkType, dataset_name=None, **kwargs):
    """A factory method for loading a custom dataset.

    Image classification datasets expect a directory of images organized with subfolders for each image class, which
    can themselves be in split directories named 'train', 'validation', and/or 'test'. Each class subfolder should
    contain .jpg images for the class. The name of the subfolder will be used as the class label.

    .. code-block:: text

        dataset_dir
          ├── class_a
          ├── class_b
          └── class_c

    Or:

    .. code-block:: text

        dataset_dir
          ├── train
          |   ├── class_a
          |   ├── class_b
          |   └── class_c
          ├── validation
          |   ├── class_a
          |   ├── class_b
          |   └── class_c
          └── test
              ├── class_a
              ├── class_b
              └── class_c

    Text classification datasets are expected to be a directory with text/csv file with two columns: the label and the
    text/sentence to classify. See the TFCustomTextClassificationDataset documentation for a list of the additional
    kwargs that are used for loading the a text classification dataset file.

    .. code-block:: text

        class_a,<text>
        class_b,<text>
        class_a,<text>
        ...

    Args:
        dataset_dir (str): directory containing the dataset
        use_case (str or UseCaseType): use case or task the dataset will be used to model
        framework (str or FrameworkType): framework
        dataset_name (str): optional; name of the dataset used for informational purposes
        kwargs: optional; additional keyword arguments depending on the type of dataset being loaded

    Returns:
        (dataset)

    Raises:
        NotImplementedError: if the type of dataset being loaded is not supported

    Example:
        >>> from tlt.datasets.dataset_factory import load_dataset
        >>> data = load_dataset('/tmp/data/flower_photos', 'image_classification', 'tensorflow')
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
                dataset_class_str = '{}.{}'.format(dataset_map[framework][use_case][dataset_catalog]['module'],
                                                   dataset_map[framework][use_case][dataset_catalog]['class'])
                dataset_class = locate(dataset_class_str)

                if not dataset_class:
                    raise NotImplementedError("Unable to find the dataset class:", dataset_class_str)
                return dataset_class(dataset_dir, dataset_name, **kwargs)

    # If no match was found in the map, then it's not implemented yet
    raise NotImplementedError("Custom dataset support for {} {} {} has not been implemented yet".format(
        str(framework), str(use_case), dataset_catalog))


def get_dataset(dataset_dir: str, use_case: UseCaseType, framework: FrameworkType,
                dataset_name: str = None, dataset_catalog: str = None, **kwargs):
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
                               datasets for PyTorch NLP models or Hugging Face models.
        **kwargs: optional; additional keyword arguments for the framework or dataset_catalog

    Returns:
        (dataset)

    Raises:
        NotImplementedError: if the dataset requested is not supported yet

    Example:
        >>> from tlt.datasets.dataset_factory import get_dataset
        >>> data = get_dataset('/tmp/data/', 'image_classification', 'tensorflow', 'tf_flowers', 'tf_datasets')
        >>> sorted(data.class_names)
        ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    """
    if not isinstance(framework, FrameworkType):
        framework = FrameworkType.from_str(framework)

    if not isinstance(use_case, UseCaseType):
        use_case = UseCaseType.from_str(use_case)

    if dataset_name and not dataset_catalog:
        # Try to assume a dataset catalog based on the other information that we have
        if framework is FrameworkType.TENSORFLOW:
            dataset_catalog = "tf_datasets"
        elif framework is FrameworkType.PYTORCH:
            if use_case in [UseCaseType.IMAGE_CLASSIFICATION, UseCaseType.OBJECT_DETECTION]:
                dataset_catalog = "torchvision"
            elif use_case is UseCaseType.TEXT_CLASSIFICATION:
                dataset_catalog = "huggingface"

    if framework in dataset_map.keys():
        if use_case in dataset_map[framework].keys():
            if dataset_catalog and dataset_catalog in dataset_map[framework][use_case]:
                dataset_class = locate('{}.{}'.format(dataset_map[framework][use_case][dataset_catalog]['module'],
                                                      dataset_map[framework][use_case][dataset_catalog]['class']))
                return dataset_class(dataset_dir, dataset_name, **kwargs)

    # If no match was found in the map, then it's not implemented yet
    raise NotImplementedError("Datasets support for {} {} {} has not been implemented yet".format(
        str(framework), str(use_case), dataset_catalog))
