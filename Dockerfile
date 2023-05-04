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

ARG IMAGE_NAME=ubuntu
ARG IMAGE_TAG=22.04
FROM ${IMAGE_NAME}:${IMAGE_TAG} as base

# TLT base target
FROM base as tlt-base

ARG PYTHON=python3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    libgl1 \
    libglib2.0-0 \
    ${PYTHON} \
    ${PYTHON}-pip && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf $(which ${PYTHON}) /usr/bin/python

# TLT target for development
FROM tlt-base as tlt-devel

ENV DEBIAN_FRONTEND=noninteractive

ENV LANG C.UTF-8
ARG PYTHON=python3

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    ${PYTHON}-dev \
    ${PYTHON}-distutils \
    build-essential \
    ca-certificates \
    make \
    pandoc && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY . /tmp/intel-transfer-learning

WORKDIR /tmp/intel-transfer-learning

RUN ${PYTHON} setup.py bdist_wheel && \
    pip install -f https://download.pytorch.org/whl/cpu/torch_stable.html --no-cache-dir dist/*.whl

# TLT target for deployment
FROM tlt-base as tlt-prod

COPY --from=tlt-devel /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=tlt-devel /usr/local/bin /usr/local/bin

ENV DATASET_DIR=/tmp/data
ENV OUTPUT_DIR=/tmp/output
