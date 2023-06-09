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

import math
import numpy as np
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion


def get_inc_config(approach='static', accuracy_criterion_relative=0.01, exit_policy_timeout=0,
                   exit_policy_max_trials=50):
    """
    Creates an INC post-training quantization config from the specified parameters.

    Args:
        static (str): Type of quantization, static or dynamic (default: static)
        accuracy_criterion_relative (float): Relative accuracy loss (default: 0.01, which is 1%)
        exit_policy_timeout (int): Tuning timeout in seconds (default: 0). Tuning processing finishes when the
                                   timeout or max_trials is reached. A tuning timeout of 0 means that the tuning
                                   phase stops when the accuracy criterion is met.
        exit_policy_max_trials (int): Maximum number of tuning trials (default: 50). Tuning processing finishes
                                      when the timeout or or max_trials is reached.

    Returns:
        A PostTrainingQuantConfig from Intel Neural Compressor

    Raises:
        ValueError: if the parameters are not within the expected values
    """
    if approach not in ['static', 'dynamic']:
        raise ValueError("Invalid value for the quantization approach ({}). Expected either "
                         "'static' or 'dynamic'.")
    if accuracy_criterion_relative and not isinstance(accuracy_criterion_relative, float) or \
            not (0.0 <= accuracy_criterion_relative <= 1.0):
        raise ValueError('Invalid value for the accuracy criterion ({}). Expected a float value between 0.0 '
                         'and 1.0'.format(accuracy_criterion_relative))
    if exit_policy_timeout and not isinstance(exit_policy_timeout, int) or exit_policy_timeout < 0:
        raise ValueError('Invalid value for the exit policy timeout ({}). Expected a positive integer or 0.'.
                         format(exit_policy_timeout))
    if exit_policy_max_trials and not isinstance(exit_policy_max_trials, int) or exit_policy_max_trials < 1:
        raise ValueError('Invalid value for max trials ({}). Expected an integer greater than 0.'.
                         format(exit_policy_timeout))

    accuracy_criterion = AccuracyCriterion(tolerable_loss=accuracy_criterion_relative)
    tuning_criterion = TuningCriterion(timeout=exit_policy_timeout, max_trials=exit_policy_max_trials)
    config = PostTrainingQuantConfig(approach=approach, device="cpu",
                                     accuracy_criterion=accuracy_criterion,
                                     tuning_criterion=tuning_criterion)

    return config


class INCTFDataLoader(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps = math.floor(len(dataset['label']) / self.batch_size)
        self.num_batch = math.ceil(self.steps / batch_size)

    def create_feed_dict_and_labels(self, dataset, batch_id=None, num_batch=None, idx=None):
        """Return the input dictionary for the given batch."""
        if idx is None:
            start_idx = batch_id * self.batch_size
            if batch_id == num_batch - 1:
                end_idx = self.steps
            else:
                end_idx = start_idx + self.batch_size
            input_ids = np.array(dataset["input_ids"])[start_idx:end_idx, :]
            attention_mask = np.array(dataset["attention_mask"])[start_idx:end_idx, :]
            feed_dict = {"input_ids": input_ids,
                         "attention_mask": attention_mask,
                         }
            labels = np.array(dataset["label"])[start_idx: end_idx]
        else:
            input_ids = np.array(dataset["input_ids"])[idx, :].reshape(1, -1)
            attention_mask = np.array(dataset["attention_mask"])[idx, :].reshape(1, -1)
            feed_dict = {"input_ids": input_ids,
                         "attention_mask": attention_mask,
                         }
            labels = np.array(dataset["label"])[idx]
        return feed_dict, labels

    def __iter__(self):
        return self.generate_dataloader(self.dataset).__iter__()

    def __len__(self):
        return self.num_batch

    def generate_dataloader(self, dataset):
        for batch_id in range(self.num_batch):
            feed_dict, labels = self.create_feed_dict_and_labels(dataset, batch_id, self.num_batch)
            yield feed_dict, labels
