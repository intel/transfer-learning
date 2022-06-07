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

from setuptools import setup, find_packages

COMMON_PACKAGES = ["click"]

EXTRA_PACKAGES = {
    "tensorflow": [
        "intel-tensorflow==2.8.0",
        "tensorflow-hub==0.12.0",
        "tensorflow-datasets==4.4.0"
    ],
    "pytorch": {
        "protobuf==3.20.1",
        "python-dateutil==2.7",
        "torch==1.11.0",
        "intel-extension-for-pytorch==1.11.0",
        "torchvision==0.12.0"
    }
}

setup(name="tlk",
      description="A Transfer Learning Kit from Intel",
      version="0.0.1",
      packages=find_packages(),
      install_requires=COMMON_PACKAGES,
      extras_require=EXTRA_PACKAGES,
      python_requires=">3.8",
      entry_points={
        "console_scripts": [
            "tlk = tlk.tools.cli.main:cli_group"
            ]
        },
      include_package_data=True
      )
