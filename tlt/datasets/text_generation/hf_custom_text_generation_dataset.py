#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
from typing import Optional

from datasets import load_dataset

from tlt.datasets.hf_dataset import HFDataset
from tlt.datasets.text_generation.text_generation_dataset import TextGenerationDataset


class HFCustomTextGenerationDataset(TextGenerationDataset, HFDataset):
    """
    A custom text generation dataset that can be used with Transformer models.
    """

    def __init__(
        self,
        dataset_dir,
        dataset_name: Optional[str],
        dataset_file: str,
        validation_file: Optional[str] = None,
        num_workers: int = 0,
        shuffle_files: bool = True,
        seed: int = None,
    ):
        """
        A custom text generation dataset that can be used with Transformer models.
        Note that this dataset class expects a .json, .txt, or .csv file with records that contain up to three keys,
        such as "instruction", "input", and "output".

        For example, a json-formatted file will look similar to the snippet below:

        .. code-block:: text
        [
            {
                "instruction": "What are the three primary colors?",
                "input": "",
                "output": "The three primary colors are red, blue, and yellow."
            },
            {
                "instruction": "Identify the odd one out.",
                "input": "Twitter, Instagram, Telegram",
                "output": "Telegram"
            },
            ...
        ]

        Args:
            dataset_dir (str): Directory containing the dataset
            dataset_name (str): Name of the dataset. If no dataset name is given, the dataset_dir folder name
                will be used as the dataset name.
            dataset_file (str): Name of the training file to load from the dataset directory; must be .json, .txt,
                                   or .csv
            validation_file (str): Optional, name of the validation file to load from the dataset directory;
                                        must be .json, .txt, or .csv
            num_workers (int): Number of workers to pass into a DataLoader.
            shuffle_files (bool): optional; Whether to shuffle the data. Defaults to True.
            seed (int): optional; Random seed for shuffling

        Raises:
            FileNotFoundError: if the file is not found in the dataset directory
        """
        train_file = os.path.join(dataset_dir, dataset_file)
        validation_file = os.path.join(dataset_dir, validation_file) if validation_file else None

        # Sanity check
        for input_file in [i for i in [train_file, validation_file] if i is not None]:
            if not os.path.exists(input_file):
                raise FileNotFoundError("The dataset file ({}) does not exist".format(input_file))

        # The dataset name is only used for informational purposes. Default to use the file name without extension.
        if not dataset_name:
            dataset_name = os.path.splitext(dataset_file)[0]

        TextGenerationDataset.__init__(self, dataset_dir, dataset_name)

        # Load the data
        extension = (
            train_file.split(".")[-1]
            if train_file is not None
            else validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"

        if train_file is not None and validation_file is not None:
            # TODO: Needs testing
            data_files = {}
            data_files["train"] = train_file
            data_files["validation"] = validation_file

            self._dataset = load_dataset(extension, data_files=data_files)
            self._validation_type = 'defined_split'
        else:
            data_files = [f for f in [train_file, validation_file] if f is not None]

            self._dataset = load_dataset(extension, data_files=data_files)['train']
            self._validation_type = None

        if shuffle_files:
            self._dataset = self._dataset.shuffle(seed=seed)

        self._info = {
            "name": dataset_name,
            "dataset_dir": dataset_dir,
            "dataset_file": dataset_file,
            "validation_file": validation_file
        }

        self._shuffle = shuffle_files
        self._num_workers = num_workers
        self._train_indices = range(len(self._dataset))
        self._validation_indices = None
        self._test_indices = None
        self._train_loader = None
        self._validation_loader = None
        self._test_loader = None
        self._preprocessed = {}

    @property
    def dataset(self):
        return self._dataset

    @property
    def info(self):
        return {'dataset_info': self._info, 'preprocessing_info': self._preprocessed}
