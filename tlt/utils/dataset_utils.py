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


def prepare_huggingface_input_data(dataset, hub_name, max_seq_length):
    """
    Prepares the input data using the BertTokenizer from Hugging Face for TensorFlow

    Args:
       dataset (TensorFlow dataset): The TensorFlow dataset to preprocess
       hub_name (str): The name of the Hugging Face model
       max_seq_length (int): The maximum sentence length to use
    Returns:
        Tokenized input and labels
    """
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(hub_name)
    data_converted = {
        'sentences': [],
        'labels': [],
    }

    for elem in dataset.as_numpy_iterator():
        # elem would be in the following format:
        # (array([sentence1, sentence2, ...]), array([label1, label2, ...]))
        data_converted['sentences'].extend(elem[0])
        data_converted['labels'].extend(elem[1])

    data_converted["sentences"] = [x.decode() for x in data_converted['sentences']]

    tokenized_dataset = tokenizer(data_converted['sentences'], padding='max_length',
                                  max_length=max_seq_length, truncation=True, return_tensors='tf')

    return tokenized_dataset, data_converted['labels']
