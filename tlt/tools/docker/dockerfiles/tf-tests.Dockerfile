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

# This dockerfile builds and installs the transfer learning CLI/API for TensorFlow
# The default command runs training based on environment variables specifying
# the model name, dataset information, output directory, etc.

ARG BASE_IMAGE="intel/intel-optimized-tensorflow"
ARG BASE_TAG="latest"

FROM ${BASE_IMAGE}:${BASE_TAG} as builder

COPY . /workspace
WORKDIR /workspace

ENV EXCLUDE_FRAMEWORK=True

RUN python setup.py bdist_wheel

FROM ${BASE_IMAGE}:${BASE_TAG}

WORKDIR /workspace
ARG TLT_VERSION=0.4.0

COPY --from=builder /workspace/dist/intel_transfer_learning_tool-${TLT_VERSION}-py3-none-any.whl .
COPY --from=builder /workspace/tests /workspace/tests

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y build-essential libgl1 libglib2.0-0 python3.9-dev && \
    pip install --upgrade pip && \
    pip install intel_transfer_learning_tool-${TLT_VERSION}-py3-none-any.whl[tensorflow] && \
    pip install tensorflow-text==2.11.0 && \
    rm intel_transfer_learning_tool-${TLT_VERSION}-py3-none-any.whl && \
    pip install -r tests/requirements-test.txt

WORKDIR /workspace

ENV PYTHONPATH=/workspace/tests

CMD ["py.test", "-s", "-m", "tensorflow or common"]
