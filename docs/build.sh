#!/usr/bin/env bash
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

COMMAND=${1}

# Temp directory within docs
TEMP_DIR="markdown"

if [[ ${COMMAND} == "clean" ]]; then
    rm -rf ${TEMP_DIR}
elif [[ ${COMMAND} == "html" ]]; then
    # Create a temp directory for markdown files that are just used for sphinx docs
    mkdir -p ${TEMP_DIR}

    # This script takes sections out of the main README.md to create smaller .md files that are used for pages
    # in the sphinx doc table of contents (like Overview, Installation, How it Works, Get Started, Legal Information).
    # If heading name changes are made in the main README.md, they will need to be updated here too because the sed
    # commands are grabbing the text between two headers.

    # We don't want to mess with the original README.md, so create a copy of it before we start editing
    cp ../README.md ${TEMP_DIR}/Welcome.md

    # Create an Overview doc
    sed -n '/^ *## Overview *$/,/^ *## Hardware Requirements *$/p' ${TEMP_DIR}/Welcome.md > ${TEMP_DIR}/Overview.md
    # Change the first instance of Intel to include the registered trademark symbol
    sed -i '0,/Intel/{s/Intel/Intel®/}' ${TEMP_DIR}/Overview.md
    sed -i '$d' ${TEMP_DIR}/Overview.md

    # Create an Installation doc (including requirements)
    echo "## Installation " > ${TEMP_DIR}/Install.md
    sed -n '/^ *## Hardware Requirements *$/,/^ *## How it Works *$/p' ${TEMP_DIR}/Welcome.md >> ${TEMP_DIR}/Install.md
    sed -i 's/## Hardware Requirements/### Hardware Requirements/g' ${TEMP_DIR}/Install.md
    sed -i '$d' ${TEMP_DIR}/Install.md
    sed -n '/^ *### Requirements *$/,/^ *### Create and activate a Python3 virtual environment *$/p' ${TEMP_DIR}/Welcome.md >> ${TEMP_DIR}/Install.md
    sed -i '$d' ${TEMP_DIR}/Install.md
    sed -i 's/### Requirements/### Software Requirements/g' ${TEMP_DIR}/Install.md
    sed -n '/^ *### Create and activate a Python3 virtual environment *$/,/^ *### Prepare the Dataset *$/p' ${TEMP_DIR}/Welcome.md >> ${TEMP_DIR}/Install.md
    sed -i '$d' ${TEMP_DIR}/Install.md
    # Change the first instance of the tool name to include the registered trademark symbol
    sed -i '0,/Intel Transfer Learning Tool/{s/Intel Transfer Learning Tool/Intel® Transfer Learning Tool/}' ${TEMP_DIR}/Install.md

    # Create a How it Works doc
    sed -n '/^ *## How it Works *$/,/^ *## Get Started *$/p' ${TEMP_DIR}/Welcome.md > ${TEMP_DIR}/HowItWorks.md
    sed -i '$d' ${TEMP_DIR}/HowItWorks.md

    # Create a Get Started doc that includes installation instruction and the CLI and API examples
    echo "# Get Started " > ${TEMP_DIR}/GetStarted.md
    sed -n '/^ *### Prepare the Dataset *$/,/^ *## Support *$/p' ${TEMP_DIR}/Welcome.md >> ${TEMP_DIR}/GetStarted.md
    sed -i 's/## Use the No-code CLI/### Use the No-code CLI/g' ${TEMP_DIR}/GetStarted.md
    sed -i 's/## Use the Low-code API/### Use the Low-code API/g' ${TEMP_DIR}/GetStarted.md
    sed -i 's/## Summary and Next Steps/### Summary and Next Steps/g' ${TEMP_DIR}/GetStarted.md
    sed -i '$d' ${TEMP_DIR}/GetStarted.md
    # Change the first instance of the tool name to include the registered trademark symbol
    sed -i '0,/Intel Transfer Learning Tool/{s/Intel Transfer Learning Tool/Intel® Transfer Learning Tool/}' ${TEMP_DIR}/GetStarted.md

    # Create a Legal Information doc
    echo "# Legal Information " > ${TEMP_DIR}/Legal.md
    sed -n '/#### DISCLAIMER: ####/,$p' ${TEMP_DIR}/Welcome.md >> ${TEMP_DIR}/Legal.md
    sed -i 's/#### DISCLAIMER: ####/#### Disclaimer/g' ${TEMP_DIR}/Legal.md
    sed -i 's/#### License: ####/#### License/g' ${TEMP_DIR}/Legal.md
    sed -i 's/#### Datasets: ####/#### Datasets/g' ${TEMP_DIR}/Legal.md
    # Change the first instance of Intel to include the registered trademark symbol
    sed -i '0,/Intel/{s/Intel/Intel®/}' ${TEMP_DIR}/Legal.md
fi

