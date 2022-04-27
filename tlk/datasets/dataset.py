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

import abc


class BaseDataset(abc.ABC):
    """
    Abstract base class for a dataset used for training or evaluation
    """
    def __init__(self, dataset_dir, dataset_name=None, dataset_catalog=None):
        self._dataset_dir = dataset_dir
        self._dataset_name = dataset_name
        self._dataset_catalog = dataset_catalog

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def dataset_dir(self):
        return self._dataset_dir

    @property
    def dataset_catalog(self):
        return self._dataset_catalog

    @property
    @abc.abstractmethod
    def dataset(self):
        pass
