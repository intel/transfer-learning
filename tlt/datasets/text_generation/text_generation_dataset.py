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

import datasets
from requests.adapters import ProxyError
import time
from transformers import AutoTokenizer

from tlt.datasets.dataset import BaseDataset


class TextGenerationDataset(BaseDataset):
    """
    Base class for a text generation dataset
    """
    def __init__(self, dataset_dir, dataset_name="", dataset_catalog=""):
        BaseDataset.__init__(self, dataset_dir, dataset_name, dataset_catalog)

    def _convert_to_prompts(self, prompt_dict, dataset_schema):
        """
            Converts the dataset to a set of prompts, with or without context, as defined by the prompt_template and
            dataset_schema.

            Args:
                prompt_dict (dict): A dictionary with keys "prompt_with_context" and/or "prompt_without_context" with
                                    which to format the raw dataset dictionaries into instruction prompts
                dataset_schema (dict): A dictionary with keys "instruction_key", "context_key", and "response_key" that
                                       maps the keys in the raw dataset dictionaries to "instruction", "context", and
                                       "response".
        """
        def create_prompts(prompt_dict, dataset_schema, examples):
            prompts = []
            for example in examples:
                if dataset_schema['context_key'] not in example.keys() or not example[dataset_schema['context_key']]:
                    prompt_template = prompt_dict["prompt_without_context"]
                else:
                    prompt_template = prompt_dict["prompt_with_context"]
                prompt = prompt_template.format_map(example)
                prompts.append(prompt)
            return prompts

        prompts = create_prompts(prompt_dict, dataset_schema, self._dataset)
        columns_to_be_removed = list(self._dataset.features.keys())
        self._dataset = self._dataset.add_column("prompts", prompts)
        self._dataset = self._dataset.remove_columns(columns_to_be_removed)

    def _concatenate_data(self, max_length=512):
        concatenated_dataset = {}
        for column in self._dataset.features:
            concatenated_data = [item for sample in self._dataset[column] for item in sample]
            reshaped_data = [concatenated_data[i * max_length:(i + 1) * max_length]
                             for i in range(len(concatenated_data) // max_length)]
            concatenated_dataset[column] = reshaped_data

        self._dataset = datasets.Dataset.from_dict(concatenated_dataset)

    def preprocess(
        self,
        model_name: str,
        batch_size: int = 8,
        prompt_dict: dict = None,
        dataset_schema: dict = None,
        max_length: int = 512,
        concatenate: bool = True
    ) -> None:
        """
        Preprocess the textual dataset to apply padding, truncation and tokenize.

            Args:
                model_name (str): Name of the model to get a matching tokenizer.
                batch_size (int): Number of examples in each batch. (default: 8)
                prompt_dict (dict): A dictionary with keys "prompt_with_context" and/or "prompt_without_context" with
                                    which to format the raw dataset dictionaries into instruction prompts
                dataset_schema (dict): A dictionary with keys "instruction_key", "context_key", and "response_key" that
                                       maps the keys in the raw dataset dictionaries to "instruction", "context", and
                                       "response".
                max_length (int): desired maximum sequence length. (default: 512)
                concatenate (bool): (default: True)

            Raises:
                ValueError: if data has already been preprocessed (or) non integer batch size given (or)
                given dataset hasn't been implemented into the API yet.
        """

        # Sanity checks
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size should be an positive integer")

        if self._preprocessed:
            raise ValueError("Data has already been preprocessed: {}".format(self._preprocessed))

        if prompt_dict:
            if not dataset_schema:
                raise ValueError("If giving a prompt_dict, please also provide a dataset_schema")
        elif dataset_schema:
            raise ValueError("If giving a dataset_schema, please also provide a prompt_dict")

        self._convert_to_prompts(prompt_dict, dataset_schema)

        # Get the tokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ProxyError:
            print("Max retries reached. Sleeping for 10 sec...")
            time.sleep(10)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Define a tokenize function to map the text to the tokenizer
        def tokenize_function(prompt, add_eos_token=True):
            results = self._tokenizer(prompt, truncation=True, max_length=max_length, padding=False,
                                      return_tensors=None)
            for i in range(len(results["input_ids"])):
                if results["input_ids"][i][-1] != self._tokenizer.eos_token_id \
                        and len(results["input_ids"][i]) < max_length \
                        and add_eos_token:
                    results["input_ids"][i].append(self._tokenizer.eos_token_id)
                    results["attention_mask"][i].append(1)

            results["labels"] = results["input_ids"].copy()

            return results

        def preprocess_function(examples):
            return tokenize_function(examples["prompts"])

        self._dataset = self._dataset.map(preprocess_function, batched=True)
        self._dataset = self._dataset.remove_columns("prompts")

        if concatenate:
            self._concatenate_data(max_length)

        # Set format to torch
        self._dataset.set_format("torch")

        self._preprocessed = {
            'max_length': max_length,
            'batch_size': batch_size,
        }
        self._make_data_loaders(batch_size=batch_size)
        print("Tokenized Dataset:", self._dataset)
