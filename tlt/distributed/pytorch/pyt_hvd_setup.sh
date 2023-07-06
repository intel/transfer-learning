#!/bin/bash
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

path_to_requirements_file=$(dirname "$(readlink -f "$0")")/requirements.txt

# Read the file line by line
while IFS= read -r package || [[ -n $package ]]; do
    # Replace "@" with "-f" using the sed command
    if [[ $package == *"@"* ]]; then
        modified_string=$(echo $package | sed 's/@/-f/g')
        pip install $modified_string
    # Install horovod with appropriate flags set
    elif [[ $package =~ "horovod" ]]; then
        HOROVOD_WITH_MPI=1
        HOROVOD_WITH_MXNET=0
        HOROVOD_WITH_PYTORCH=1
        HOROVOD_WITH_TENSORFLOW=0
        pip install --no-cache-dir $package
    else
        pip install $package
    fi
done < $path_to_requirements_file
