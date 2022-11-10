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
import random

from transformers import AutoTokenizer

from torch.utils.data import DataLoader as loader

from tlt.datasets.dataset import BaseDataset


class HFDataset(BaseDataset):
    """
    Base class used to represent Hugging Face Dataset
    """

    def __init__(self, dataset_dir, dataset_name="", dataset_catalog=""):
        BaseDataset.__init__(dataset_dir, dataset_name, dataset_catalog)

    def get_batch(self, subset='all'):
        """
        Get a single batch of images and labels from the dataset.

            Args:
                subset (str): default "all", can also be "train", "validation", or "test"

            Returns:
                (examples, labels)

            Raises:
                ValueError if the dataset is not defined yet or the given subset is not valid
        """

        if subset == 'all' and self._dataset is not None:
            return next(iter(self._data_loader))
        elif subset == 'train' and self.train_subset is not None:
            return next(iter(self._train_loader))
        elif subset == 'validation' and self.validation_subset is not None:
            return next(iter(self._validation_loader))
        elif subset == 'test' and self.test_subset is not None:
            return next(iter(self._test_loader))
        else:
            raise ValueError("Unable to return a batch, because the dataset or subset hasn't been defined.")

    def preprocess(
        self,
        model_name: str,
        batch_size: int = 32,
        padding: str = "max_length",
        truncation: bool = True
    ) -> None:
        """
        Preprocess the textual dataset to apply padding, truncation and tokenize.

            Args:
                model_name (str): Name of the model to get a matching tokenizer.
                batch_size (int): Number of batches to split the data.
                padding (str): desired padding. (default: "max_length")
                truncation (bool): Boolean specifying to truncate the word tokens to match with the
                longest sentence. (default: True)
            Raises:
                ValueError if data has already been preprocessed (or) non integer batch size given (or)
                given dataset hasn't been implemented into the API yet.
        """

        # Sanity checks
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size should be an positive integer")

        if self._preprocessed:
            raise ValueError("Data has already been preprocessed: {}".format(self._preprocessed))

        # Get the column names of the textual data for the tokenizer
        dataset_columns = self._dataset.column_names
        text_column_name = None
        text_column_name_2 = None

        if 'text' in dataset_columns:
            text_column_name = 'text'
        elif 'sentence' in dataset_columns:
            text_column_name = 'sentence'
        elif 'sentence1' and 'sentence2' in dataset_columns:
            text_column_name = 'sentence1'
            text_column_name_2 = 'sentence2'

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(examples):
            # Define the tokenizer args, depending on if the data has 2 textual columns or just 1
            args = ((examples[text_column_name],) if text_column_name_2 is None
                    else (examples[text_column_name], examples[text_column_name_2]))
            return self._tokenizer(*args, padding=padding, truncation=truncation)

        self._dataset = self._dataset.map(tokenize_function, batched=True)

        # Remove the raw text from the tokenized dataset
        raw_text_columns = [text_column_name, text_column_name_2] if text_column_name_2 else [text_column_name]
        self._dataset = self._dataset.remove_columns(raw_text_columns)

        self._preprocessed = {
            'padding': padding,
            'truncation': truncation,
            'batch_size': batch_size,
        }
        self._make_data_loaders(batch_size=batch_size)
        print("tokenized_dataset:", self._dataset)

    def shuffle_split(self, train_pct=.75, val_pct=.25, test_pct=0., seed=None):

        # Sanity checks
        if not (isinstance(train_pct, float) and isinstance(val_pct, float) and isinstance(test_pct, float)):
            raise ValueError("Percentage arguments must be floats.")

        if train_pct + val_pct + test_pct > 1.0:
            raise ValueError("Sum of percentage arguments must be less than or equal to 1.")

        self._validation_type = 'shuffle_split'

        # Calculating splits
        length = len(self._dataset)
        train_size = int(train_pct * length)
        val_size = int(val_pct * length)
        test_size = int(test_pct * length)

        generator = torch.Generator().manual_seed(seed) if seed else None
        dataset_indices = torch.randperm(length, generator=generator).tolist()
        self._train_indices = dataset_indices[:train_size]
        self._validation_indices = dataset_indices[train_size:train_size + val_size]

        if test_pct:
            self._test_indices = dataset_indices[train_size + val_size:train_size + val_size + test_size]
        else:
            self._test_indices = None

        if self._preprocessed and 'batch_size' in self._preprocessed:
            self._make_data_loaders(batch_size=self._preprocessed['batch_size'], generator=generator)

        print("Dataset split into:")
        print("-------------------")
        print("{} train samples".format(train_size))
        print("{} test samples".format(test_size))
        print("{} validation samples".format(val_size))

    def _make_data_loaders(self, batch_size, generator=None):

        def seed_worker(worker_id):
            import numpy as np
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        if self._validation_type == 'shuffle_split':
            self._train_loader = loader(self.train_subset, batch_size=batch_size, shuffle=self._shuffle,
                                        num_workers=self._num_workers, worker_init_fn=seed_worker, generator=generator)

            self._validation_loader = loader(self.validation_subset, batch_size=batch_size, shuffle=self._shuffle,
                                             num_workers=self._num_workers, worker_init_fn=seed_worker,
                                             generator=generator)

            if self._test_indices:
                self._test_loader = loader(self.test_subset, batch_size=batch_size, shuffle=self._shuffle,
                                           num_workers=self._num_workers, worker_init_fn=seed_worker,
                                           generator=generator)

        elif self._validation_type == 'defined_split':
            if 'train' in self._split:
                self._train_loader = loader(self.train_subset, batch_size=batch_size, shuffle=self._shuffle,
                                            num_workers=self._num_workers, worker_init_fn=seed_worker,
                                            generator=generator)
            if 'test' in self._split:
                self._test_loader = loader(self.test_subset, batch_size=batch_size, shuffle=self._shuffle,
                                           num_workers=self._num_workers, worker_init_fn=seed_worker,
                                           generator=generator)
            if 'validation' in self._split:
                self._validation_loader = loader(self.validation_subset, batch_size=batch_size, shuffle=self._shuffle,
                                                 num_workers=self._num_workers, worker_init_fn=seed_worker,
                                                 generator=generator)
        elif self._validation_type == 'recall':
            self._data_loader = loader(self._dataset, batch_size=batch_size, shuffle=self._shuffle,
                                       num_workers=self._num_workers, worker_init_fn=seed_worker, generator=generator)

            self._train_loader = self._data_loader
            self._test_loader = self._data_loader
            self._validation_loader = self._data_loader

    @property
    def train_loader(self):
        if self._train_loader:
            return self._train_loader
        else:
            raise ValueError("train split not specified")

    @property
    def test_loader(self):
        if self._test_loader:
            return self._test_loader
        else:
            raise ValueError("test split not specified")

    @property
    def validation_loader(self):
        if self._validation_loader:
            return self._validation_loader
        else:
            raise ValueError("validation split not specified")
