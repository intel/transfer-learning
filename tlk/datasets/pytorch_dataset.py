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

import torch

from tlk.datasets.dataset import BaseDataset


class PyTorchDataset(BaseDataset):
    """
    Class used to represent a PyTorch Dataset
    """

    def __init__(self, dataset_dir, dataset_name="", dataset_catalog=""):
        BaseDataset.__init__(self, dataset_dir, dataset_name, dataset_catalog)

    @property
    def train_subset(self):
        return torch.utils.data.Subset(self._dataset, self._train_indices) if self._train_indices else None

    @property
    def validation_subset(self):
        return torch.utils.data.Subset(self._dataset, self._validation_indices) if self._validation_indices else None

    @property
    def test_subset(self):
        return torch.utils.data.Subset(self._dataset, self._test_indices) if self._test_indices else None

    @property
    def data_loader(self):
        return self._data_loader

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def validation_loader(self):
        return self._validation_loader

    @property
    def test_loader(self):
        return self._test_loader

    def get_batch(self, subset='all'):
        """Get a single batch of images and labels from the dataset.

            Args:
                subset (str): default "all", can also be "train", "validation", or "test"

            Returns:
                (examples, labels)

            Raises:
                ValueError if the dataset is not defined yet or the given subset is not valid
        """
        if subset == 'all' and self._dataset is not None:
            return next(iter(self._data_loader))
        elif subset == 'train' and self._train_subset is not None:
            return next(iter(self._train_loader))
        elif subset == 'validation' and self._validation_subset is not None:
            return next(iter(self._validation_loader))
        elif subset == 'test' and self._test_subset is not None:
            return next(iter(self._test_loader))
        else:
            raise ValueError("Unable to return a batch, because the dataset or subset hasn't been defined.")

    def shuffle_split(self, train_pct=.75, val_pct=.25, test_pct=0., seed=None):
        """Randomly splits the dataset into train, validation, and test subsets with a pseudo-random seed option.

            Args:
                train_pct (float): default .75, percentage of dataset to use for training
                val_pct (float):  default .25, percentage of dataset to use for validation
                test_pct (float): default 0.0, percentage of dataset to use for testing
                seed (None or int): default None, can be set for pseudo-randomization

            Raises:
                ValueError if percentage input args are not floats or sum to greater than 1
        """
        if not (isinstance(train_pct, float) and isinstance(val_pct, float) and isinstance(test_pct, float)):
            raise ValueError("Percentage arguments must be floats.")
        if train_pct + val_pct + test_pct > 1.0:
            raise ValueError("Sum of percentage arguments must be less than or equal to 1.")

        length = len(self._dataset)
        train_size = int(train_pct * length)
        val_size = int(val_pct * length)
        test_size = int(test_pct * length)
        generator = torch.Generator().manual_seed(seed) if seed else None
        dataset_indices = torch.randperm(length, generator=generator).tolist()
        self._train_indices = dataset_indices[:train_size]
        self._validation_indices = dataset_indices[train_size:train_size+val_size]
        if test_pct:
            self._test_indices = dataset_indices[train_size+val_size:train_size+val_size+test_size]
        else:
            self._test_indices = None
        self._validation_type = 'shuffle_split'
        if self._preprocessed and 'batch_size' in self._preprocessed:
            self._make_data_loaders(batch_size=self._preprocessed['batch_size'])
