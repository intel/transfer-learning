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
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader as loader

from tlk.datasets.pytorch_dataset import PyTorchDataset
from tlk.datasets.image_classification.image_classification_dataset import ImageClassificationDataset

DATASETS = ["CIFAR10", "Flowers102", "Food101", "GTSRB"]

class TorchvisionImageClassificationDataset(ImageClassificationDataset, PyTorchDataset):
    """
    Base class for an image classification dataset from the torchvision catalog
    """
    def __init__(self, dataset_dir, dataset_name, split=['train'], download=True, num_workers=0):
        if not isinstance(split, list):
            raise ValueError("Value of split argument must be a list.")
        for s in split:
            if not isinstance(s, str) or s not in ['train', 'validation', 'test']:
                raise ValueError('Split argument can only contain these strings: train, validation, test.')
        if dataset_name not in DATASETS:
            raise ValueError("Dataset name is not supported. Choose from: {}".format(DATASETS))
        else:
            dataset_class = locate('torchvision.datasets.{}'.format(dataset_name))
        ImageClassificationDataset.__init__(self, dataset_dir, dataset_name)
        self._num_workers = num_workers
        self._preprocessed = {}
        self._dataset = None
        self._train_indices = None
        self._validation_indices = None
        self._test_indices = None

        if len(split) == 1:
            # If there is only one split, use it for _dataset and do not define any indices
            if split[0] == 'train':
                try:
                    self._dataset = dataset_class(dataset_dir, split='train', download=True)
                except:
                    self._dataset = dataset_class(dataset_dir, train=True, download=True)
            elif split[0] == 'validation':
                try:
                    self._dataset = dataset_class(dataset_dir, split='val', download=True)
                except:
                    raise ValueError('No validation split was found for this dataset: {}'.format(dataset_name))
            elif split[0] == 'test':
                try:
                    self._dataset = dataset_class(dataset_dir, split='test', download=True)
                except:
                    try:
                        self._dataset = dataset_class(dataset_dir, train=False, download=True)
                    except:
                        raise ValueError('No test split was found for this dataset: {}'.format(dataset_name))
            self._validation_type = 'recall'  # Train & evaluate on the whole dataset
        else:
            # If there are multiple splits, concatenate them for _dataset and define indices
            if 'train' in split:
                try:
                    self._dataset = dataset_class(dataset_dir, split='train', download=True)
                except:
                    self._dataset = dataset_class(dataset_dir, train=True, download=True)
                self._train_indices = range(len(self._dataset))
            if 'validation' in split:
                try:
                    validation_data = dataset_class(dataset_dir, split='val', download=True)
                    validation_length = len(validation_data)
                    if self._dataset:
                        current_length = len(self._dataset)
                        self._dataset = torch.utils.data.ConcatDataset([self._dataset, validation_data])
                        self._validation_indices = range(current_length, current_length+validation_length)
                    else:
                        self._dataset = validation_data
                        self._validation_indices = range(validation_length)
                except:
                    raise ValueError('No validation split was found for this dataset: {}'.format(dataset_name))
            if 'test' in split:
                try:
                    test_data = dataset_class(dataset_dir, split='test', download=True)
                except:
                    try:
                        test_data = dataset_class(dataset_dir, train=False, download=True)
                    except:
                        raise ValueError('No test split was found for this dataset: {}'.format(dataset_name))
                finally:
                    test_length = len(test_data)
                    if self._dataset:
                        current_length = len(self._dataset)
                        self._dataset = torch.utils.data.ConcatDataset([self._dataset, test_data])
                        self._test_indices = range(current_length, current_length+test_length)
                    else:
                        self._dataset = test_data
                        self._validation_indices = range(test_length)
            self._validation_type = 'defined_split'  # Defined by user or torchvision
        self._make_data_loaders(batch_size=1)
        self._info = {'name': dataset_name, 'size': len(self._dataset)}

    def _make_data_loaders(self, batch_size):
        """Make data loaders for the whole dataset and the subsets that have indices defined"""
        if self._dataset:
            self._data_loader = loader(self.dataset, batch_size=batch_size,
                                       shuffle=False, num_workers=self._num_workers)
        else:
            self._data_loader = None
        if self._train_indices:
            self._train_loader = loader(self.train_subset, batch_size=batch_size,
                                        shuffle=False, num_workers=self._num_workers)
        else:
            self._train_loader = None
        if self._validation_indices:
            self._validation_loader = loader(self.validation_subset, batch_size=batch_size,
                                             shuffle=False, num_workers=self._num_workers)
        else:
            self._validation_loader = None
        if self._test_indices:
            self._test_loader = loader(self.test_subset, batch_size=batch_size, shuffle=False,
                                       num_workers=self._num_workers)
        else:
            self._test_loader = None

    @property
    def class_names(self):
        return self._dataset.classes

    @property
    def info(self):
        return {'dataset_info': self._info, 'preprocessing_info': self._preprocessed}

    @property
    def dataset(self):
        return self._dataset

    def preprocess(self, image_size, batch_size):
        """Preprocess the dataset to resize, normalize, and batch the images

            Args:
                image_size (int): desired square image size
                batch_size (int): desired batch size

            Raises:
                ValueError if the dataset is not defined or has already been processed
        """
        # NOTE: Should this be part of init? If we get image_size and batch size during init,
        # then we don't need a separate call to preprocess.
        def get_transform(image_size):
            transforms = []
            transforms.append(T.Resize([image_size, image_size]))
            transforms.append(T.ToTensor())
            transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

            return T.Compose(transforms)

        if not (self._dataset):
            raise ValueError("Unable to preprocess, because the dataset hasn't been defined.")
        if self._preprocessed and image_size != self._preprocessed['image_size']:
            raise ValueError("Data has already been preprocessed with a different image size: {}".
                             format(self._preprocessed))

        self._dataset.transform = get_transform(image_size)
        self._make_data_loaders(batch_size=batch_size)
        self._preprocessed = {'image_size': image_size, 'batch_size': batch_size}
