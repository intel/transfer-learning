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

import os

from pathlib import Path
from setuptools import setup, find_packages

COMMON_PACKAGES = ["click"]

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

def get_framework_requirements(framework_name):
    """ Gets framework related requirements from its requirements.txt file """
    with open('{}_requirements.txt'.format(framework_name)) as f:
        requirements = f.read().splitlines()

    if os.environ.get("EXCLUDE_FRAMEWORK", default="False") == "True":
        # items to exclude if we don't want to install the framework
        exclude_list = ["tensorflow", "intel-tensorflow", "torch", "intel-extension-for-pytorch", "torchvision"]
        requirements = [r for r in requirements if r.split('=')[0] not in exclude_list]

    return requirements

EXTRA_PACKAGES = {
    "tensorflow": get_framework_requirements("tensorflow"),
    "pytorch": get_framework_requirements("pytorch")
}

setup(name="intel-transfer-learning-tool",
      description="Intel® Transfer Learning Tool",
      version="0.5.0",
      url='https://github.com/IntelAI/transfer-learning',
      license='Apache 2.0',
      author='IntelAI',
      author_email='IntelAI@intel.com',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires= \
        COMMON_PACKAGES + \
        EXTRA_PACKAGES['pytorch'] + \
        EXTRA_PACKAGES['tensorflow'],
      extras_require=EXTRA_PACKAGES,
      python_requires=">3.8",
      entry_points={
        "console_scripts": [
            "tlt = tlt.tools.cli.main:cli_group"
            ]
        },
      include_package_data=True
      )
