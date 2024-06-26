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


ARG IMAGE_NAME=intel/intel-optimized-pytorch
ARG IMAGE_TAG=2.2.0-pip-multinode

# Base image to be used everywhere
FROM ${IMAGE_NAME}:${IMAGE_TAG} as base
RUN apt-get clean && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Base Image has python installed
ARG PYTHON=python
ARG PYTHON_VER=3.10

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    google-perftools \
    libjemalloc2 \
    libomp-dev \
    numactl

RUN python -m pip install --no-cache-dir \
    'SentencePiece==0.2.0' \
    'accelerate==0.28.0' \
    'datasets==2.19.0' \
    'einops==0.7.0' \
    'evaluate==0.4.1' \
    'mkl-include==2023.2.0' \
    'mkl==2023.2.0' \
    'nltk==3.8.1' \
    'onnxruntime-extensions==0.10.1' \
    'onnxruntime==1.17.3' \
    'peft==0.10.0' \
    'protobuf==4.24.4' \
    'psutil==5.9.5' \
    'py-cpuinfo==9.0.0' \
    'rouge_score==0.1.2' \
    'tokenizers==0.19.1'

# Install OpenSSH for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    openssh-client \
    openssh-server && \
    rm  /etc/ssh/ssh_host_*_key \
    /etc/ssh/ssh_host_*_key.pub && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Allow OpenSSH to talk to containers without asking for confirmation
RUN mkdir -p /var/run/sshd && \
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

COPY scripts /workspace/scripts/

COPY generate_ssh_keys.sh /workspace/generate_ssh_keys.sh

RUN cat /workspace/generate_ssh_keys.sh >> ${HOME}/.bash_profile && \
    sed -i 's#source /inc/bin/activate##g' ${HOME}/.bashrc && \
    cat /workspace/generate_ssh_keys.sh >> ${HOME}/.bashrc

ENV BASH_ENV=${HOME}/.bash_profile

# Set CCL env vars
ENV PATH="/usr/local/lib/${PYTHON}${PYTHON_VER}/dist-packages/oneccl_bindings_for_pytorch/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib/${PYTHON}${PYTHON_VER}/dist-packages/oneccl_bindings_for_pytorch/lib:${LD_LIBRARY_PATH}"
