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


ARG BASE_IMAGE=python
ARG BASE_TAG=3.9
FROM ${BASE_IMAGE}:${BASE_TAG} as python-base

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates \
    gawk \
    git \
    git-lfs \
    google-perftools \
    iproute2 \
    libgl1 \
    libjemalloc2 \
    numactl \
    wget && \
    git lfs install    

ARG ONECCL_VER=2.0.0+cpu
ARG TORCH_VER=2.0.1+cpu
ARG IPEX_VER=2.0.100
ARG INC_VER=2.3

RUN python -m pip install --no-cache-dir oneccl_bind_pt==${ONECCL_VER} \
        -f https://developer.intel.com/ipex-whl-stable-cpu && \
    python -m pip install --no-cache-dir torch==${TORCH_VER} \
    --extra-index-url https://download.pytorch.org/whl/cpu --force-reinstall && \
    python -m pip install --no-cache-dir neural-compressor==${INC_VER} \
        intel_extension_for_pytorch==${IPEX_VER} \
        onnx \
        onnxruntime \
        onnxruntime-extensions \
        git+https://github.com/huggingface/optimum-intel.git \
        evaluate

RUN python -m pip install --no-cache-dir --upgrade \
        accelerate \
        datasets \
        einops \
        evaluate \
        mkl==2023.2.0 \
        mkl-include==2023.2.0 \
        nltk \
        numpy \
        peft \
        pip \
        protobuf==3.20.1 \
        psutil \
        py-cpuinfo \
        rouge_score \
        SentencePiece \
        setuptools \
        tokenizers

# Install OpenSSH for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    libopenmpi-dev \
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

COPY generate_ssh_keys /workspace/generate_ssh_keys

RUN cat /workspace/generate_ssh_keys >> ${HOME}/.bash_profile && \
    cat /workspace/generate_ssh_keys >> ${HOME}/.bashrc

ENV BASH_ENV=${HOME}/.bash_profile
