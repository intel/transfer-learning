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

ENV LANG C.UTF-8

# TLT base target
FROM base as tlt-base

ARG PYTHON=python3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    libgl1 \
    libglib2.0-0 \
    ${PYTHON} \
    python3-pip && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf "$(which ${PYTHON})" /usr/bin/python

# TLT target for GitHub actions
FROM tlt-base as tlt-ci

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

RUN ${PYTHON} -m pip install virtualenv

# TLT target for development
FROM tlt-ci as tlt-devel

ARG PYTHON=python3

COPY . /tmp/intel-transfer-learning

WORKDIR /tmp/intel-transfer-learning

RUN ${PYTHON} setup.py bdist_wheel && \
    ${PYTHON} -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu dist/*.whl

# TLT target for deployment
FROM tlt-base as tlt-prod

COPY --from=tlt-devel /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=tlt-devel /usr/local/bin /usr/local/bin

ENV DATASET_DIR=/tmp/data
ENV OUTPUT_DIR=/tmp/output

# TLT target for running with MPI
FROM tlt-prod as tlt-mpi

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    openssh-client \
    openssh-server && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/bin/mpirun /usr/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/bin/mpirun && \
    chmod a+x /usr/bin/mpirun

# Configure OpenMPI to run good defaults:
RUN echo "btl_tcp_if_exclude = lo,docker0" >> /etc/openmpi/openmpi-mca-params.conf

# Install OpenSSH for MPI to communicate between containers and allow OpenSSH to
# talk to containers without asking for confirmation
RUN mkdir -p /var/run/sshd && \
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# TLT target for with MPI, Horovod and all development tools
FROM tlt-mpi as tlt-dist-devel

ARG ONECCL_VERSION=2.1.0

RUN python -m pip install --no-cache-dir https://intel-extension-for-pytorch.s3.amazonaws.com/torch_ccl/cpu/oneccl_bind_pt-${ONECCL_VERSION}%2Bcpu-cp310-cp310-linux_x86_64.whl

FROM tlt-mpi as tlt-dist-prod

ARG PYTHON=python3.10

COPY --from=tlt-dist-devel /usr/local/lib/${PYTHON}/dist-packages /usr/local/lib/${PYTHON}/dist-packages
COPY --from=tlt-dist-devel /usr/local/bin /usr/local/bin

FROM tlt-mpi as tlt-dist-devel-hvd

ARG PYTHON=python3
ARG HOROVOD_WITH_PYTORCH=1
ARG HOROVOD_WITHOUT_MXNET=1
ARG HOROVOD_WITH_TENSORFLOW=1
ARG HOROVOD_VERSION=0.28.1
ARG TORCH_AUDIO_VERSION=2.0.2
ARG TORCH_VERSION=2.0.1
ARG TORCH_VISION_VERSION=0.15.2
ARG TORCH_CPU_URL=https://download.pytorch.org/whl/cpu
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    cmake \
    g++ \
    gcc \
    git \
    ${PYTHON}-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY tensorflow_requirements.txt /tmp/tensorflow_requirements.txt

RUN python -m pip install --no-cache-dir --extra-index-url ${TORCH_CPU_URL} \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCH_AUDIO_VERSION} \
    torchvision==${TORCH_VISION_VERSION} \
    pip install --no-cache-dir -r /tmp/tensorflow_requirements.txt && \
    python -m pip install --no-cache-dir horovod[mpi,pytorch,tensorflow]==${HOROVOD_VERSION}

FROM tlt-mpi as tlt-dist-prod-hvd

ARG PYTHON=python3.10

COPY --from=tlt-dist-devel-hvd /usr/local/lib/${PYTHON}/dist-packages /usr/local/lib/${PYTHON}/dist-packages
COPY --from=tlt-dist-devel-hvd /usr/local/bin /usr/local/bin
