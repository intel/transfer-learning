#!/usr/bin/env bash
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

# Var to specify to build a unit test container (default to False)
TEST_CONTAINER=${TEST_CONTAINER:-False}

# TLK_VERSION used in the name of the wheel that's installed
TLK_VERSION=${TLK_VERSION:-0.0.1}

if [[ -z "${FRAMEWORK}" ]]; then
    echo "The FRAMEWORK environment variable is undefined. Set FRAMEWORK to build tlk for either tensorflow or pytorch"
    exit 1
fi

if [[ "${FRAMEWORK,,}" == "tensorflow" ]]; then
    if [[ "${TEST_CONTAINER,,}" == "true" ]]; then
        echo "Building tlk unit test container for TensorFlow"
        DOCKERFILE=tf-tests.Dockerfile
        IMAGE_NAME=${IMAGE_NAME:-intel/tlk:tensorflow-tests}
    else
        echo "Building tlk container for TensorFlow"
        DOCKERFILE=tf.Dockerfile
        IMAGE_NAME=${IMAGE_NAME:-intel/tlk:tensorflow}
    fi

    BASE_IMAGE=${BASE_IMAGE:-intel/intel-optimized-tensorflow}
    BASE_TAG=${BASE_TAG:-latest}
elif [[ "${FRAMEWORK,,}" == "pytorch" ]]; then
    if [[ "${TEST_CONTAINER,,}" == "true" ]]; then
        echo "Building tlk unit test container for PyTorch"
        DOCKERFILE=pyt-tests.Dockerfile
        IMAGE_NAME=${IMAGE_NAME:-intel/tlk:pytorch-tests}
    else
        echo "Building tlk container for PyTorch"
        DOCKERFILE=pyt.Dockerfile
        IMAGE_NAME=${IMAGE_NAME:-intel/tlk:pytorch}
    fi

    BASE_IMAGE=${BASE_IMAGE:-intel/intel-optimized-pytorch}
    BASE_TAG=${BASE_TAG:-latest}
else
    echo "Unrecognized FRAMEWORK value '${FRAMEWORK}' (expected 'tensorflow' or 'pytorch')"
    exit 1
fi

# If the TLK_REPO env var hasn't been set, warn the user and default to use the current directory
if [[ -z "${TLK_REPO}" ]]; then
    echo "The TLK_REPO environment variable is undefined. Setting TLK_REPO to the current directory."
    TLK_REPO=$(pwd)
fi

echo "FRAMEWORK: ${FRAMEWORK}"
echo "TEST_CONTAINER: ${TEST_CONTAINER}"
echo "BASE_IMAGE: ${BASE_IMAGE}"
echo "BASE_TAG: ${BASE_TAG}"
echo "DOCKERFILE: $DOCKERFILE"
echo "IMAGE_NAME: $IMAGE_NAME"
echo "TLK_REPO: ${TLK_REPO}"
echo "TLK_VERSION: ${TLK_VERSION}"

TLK_DOCKERFILE_PATH=${TLK_REPO}/tlk/tools/docker/dockerfiles/${DOCKERFILE}
if [[ ! -f ${TLK_DOCKERFILE_PATH} ]]; then
    # Error if the dockerfile can't be found
    echo "The dockerfile could not be found at: ${TLK_DOCKERFILE_PATH}"
    echo "Please ensure that the TLK_REPO environment variable is pointing to the repo's root directory."
    exit 1
fi

# Temporarily copy the dockerfile to the tlk root so that we can build the image
cp ${TLK_DOCKERFILE_PATH} ${TLK_REPO}/${DOCKERFILE}
pushd ${TLK_REPO}

# Build the image
docker build --no-cache \
             --build-arg http_proxy=$http_proxy \
             --build-arg https_proxy=$https_proxy  \
             --build-arg no_proxy=$no_proxy \
             --build-arg BASE_IMAGE=${BASE_IMAGE} \
             --build-arg BASE_TAG=${BASE_TAG} \
             --build-arg TLK_VERSION=${TLK_VERSION} \
             -t ${IMAGE_NAME} \
             -f ${DOCKERFILE} .

# Remove the temp copy of the dockerfile
rm ${TLK_REPO}/${DOCKERFILE}

popd
