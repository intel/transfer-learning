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

ARG PYTHON

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    libgl1 \
    libglib2.0-0 \
    ${PYTHON} \
    python3-pip

RUN ln -sf $(which ${PYTHON}) /usr/bin/python

# TLT target for development
FROM tlt-base as base-devel

ENV LANG C.UTF-8

RUN apt-get install -y --no-install-recommends --fix-missing \
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
    pip install --no-cache-dir -f https://developer.intel.com/ipex-whl-stable-cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html dist/*.whl

# TLT target for deployment
FROM tlt-base as base-prod

COPY --from=base-devel /usr/local/lib/${PYTHON}/dist-packages /usr/local/lib/${PYTHON}/dist-packages
COPY --from=base-devel /usr/local/bin /usr/local/bin

ENV DATASET_DIR=/tmp/data
ENV OUTPUT_DIR=/tmp/output

FROM base-prod as mpi

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    openssh-client \
    openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/bin/mpirun /usr/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/bin/mpirun && \
    chmod a+x /usr/bin/mpirun

# Configure OpenMPI to run good defaults:
RUN echo "btl_tcp_if_exclude = lo,docker0" >> /etc/openmpi/openmpi-mca-params.conf

# Install OpenSSH for MPI to communicate between containers
RUN mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

FROM mpi as dist-devel

ARG HOROVOD_WITH_PYTORCH=1
ARG HOROVOD_WITHOUT_MXNET=1
ARG HOROVOD_WITH_TENSORFLOW=1
ARG HOROVOD_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    cmake \
    g++ \
    gcc \
    git \
    ${PYTHON}-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir horovod==${HOROVOD_VERSION}

ARG ONECCL_VERSION
ARG ONECCL_URL=https://developer.intel.com/ipex-whl-stable-cpu

RUN python -m pip install --no-cache-dir oneccl_bind_pt==${ONECCL_VERSION} -f ${ONECCL_URL}

COPY . /tmp/intel-transfer-learning

WORKDIR /tmp/intel-transfer-learning

FROM mpi as dist-prod

COPY --from=dist-devel /usr/local/lib/${PYTHON}/dist-packages /usr/local/lib/${PYTHON}/dist-packages
COPY --from=dist-devel /usr/local/bin /usr/local/bin
