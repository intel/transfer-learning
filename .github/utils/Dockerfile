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

ARG IMAGE_NAME=ubuntu
ARG IMAGE_TAG=20.04
FROM ${IMAGE_NAME}:${IMAGE_TAG} as devel-build

ENV LANG C.UTF-8
ARG PYTHON=python3.9

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    libgl1 \
    libglib2.0-0 \
    make \
    pandoc \
    ${PYTHON} \
    ${PYTHON}-dev \
    ${PYTHON}-distutils

RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/bin/python3

RUN curl -fSsL https://bootstrap.pypa.io/get-pip.py | ${PYTHON}
