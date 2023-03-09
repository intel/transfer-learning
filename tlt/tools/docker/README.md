# Docker

Follow the instructions below to build and run `tlt` using a docker container.

## Building containers

There are currently 4 different containers that can be built:
* Intel® Transfer Learning Tool (`tlt`) for TensorFlow
* Intel® Transfer Learning Tool (`tlt`) for PyTorch
* Intel® Transfer Learning Tool (`tlt`) unit testing for TensorFlow
* Intel® Transfer Learning Tool (`tlt`) unit testing for PyTorch

There's a [`build.sh`](build.sh) script that will build a container based on environment
variables that specify which framework, base container, etc to use.

### Prerequisites

Before running these instructions, install the following:
* Docker
* git

### Variables

The environment variables in the table below can be set before running `build.sh`.
If they are not set, their default value will be used.

| Environment variable | Description | Default value |
|----------------------|-------------|---------------|
| `FRAMEWORK` (required) | Specify to build the image for TensorFlow or PyTorch. Set `FRAMEWORK` to `tensorflow` or `pytorch`. | None |
| `TEST_CONTAINER` | Specify to build a container that runs unit tests. | `False` |
| `TLT_REPO` | Path to the root directory of the cloned repository. | `$(pwd)` (the current directory) |
| `IMAGE_NAME` | The name/tag of the image being built. | `intel/tlt:tensorflow` or `intel/tlt:pytorch`, depending on `FRAMEWORK`. Unit test containers are named `intel/tlt:tensorflow-tests` and `intel/tlt:pytorch-tests`  |
| `BASE_IMAGE` | The base image name. | `intel/intel-optimized-tensorflow` or `intel/intel-optimized-pytorch`, depending on `FRAMEWORK` |
| `BASE_TAG` | The tag for the base image. | `latest` |

### Running the build script
Build the docker container using the `build.sh` script and dockerfiles from this repository:
```
# Clone the git repo, if you don't already have it
git clone https://github.com/IntelAI/transfer-learning.git
cd transfer-learning

# Build the tensorflow tlt container
FRAMEWORK=tensorflow tlt/tools/docker/build.sh

# Or build the pytorch tlt container
FRAMEWORK=pytorch tlt/tools/docker/build.sh

# Or, build a unit test container by setting TEST_CONTAINER=True
FRAMEWORK=tensorflow TEST_CONTAINER=True tlt/tools/docker/build.sh
```

## Run the container

### CLI commands in the container

`tlt` commands can be run in the container. The example below is using the tensorflow
container, but the same commands can be run using the pytorch container, by
changing the image name/tag to `intel/tlt:pytorch`.

For example, to list the available TensorFlow models:
```
docker run --rm \
    -u $(id -u):$(id -g) \
    -it intel/tlt:tensorflow \
    tlt list models -f tensorflow
```

You can also run the container interactively. The example below shows how to do this
while mounting volumes for an output and dataset directory.
```
OUTPUT_DIR=<path to an output directory>
DATASET_DIR=<path to your dataset>

docker run --rm \
    -u $(id -u):$(id -g) \
    --env http_proxy=${http_proxy} \
    --env https_proxy=${https_proxy} \
    --env OUTPUT_DIR=${OUTPUT_DIR} \
    --env DATASET_DIR=${DATASET_DIR} \
    -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
    -v ${DATASET_DIR}:${DATASET_DIR} \
    -it intel/tlt:tensorflow \
    /bin/bash
```
After executing this command, you will be at a prompt in the container where
you can execute the `tlt` CLI or start a python session to use the `tlt` API.

### Training

By default, the container that will run `tlt train`. Provide environment
variables to specify parameters like the name of the model to run, dataset, and
an output directory.

| Environment variable | Description |
|----------------------|-------------|
| `MODEL_NAME` | Name of the model to run. |
| `DATASET_DIR` | The dataset directory path for a custom dataset, or if a `DATASET_NAME` is being provided, the dataset directory is the location where the named dataset will be downloaded. |
| `DATASET_NAME` | If a custom dataset is not being used, provide the name of a dataset from a dataset catalog (TFDS or torchvision), for example `tf_flowers`. |
| `OUTPUT_DIR` | Directory where logs, checkpoints, and saved models will be written. |
| `EPOCHS` | The number of training epochs to run. Defaults to `1`. |

The snippet below shows an example of how to run transfer learning using `efficientnet_b0`
with a dataset on your local machine. The output and dataset directories are
mounted in the container and environment variables for the specified parameters
are being passed to the container. If you have other environment variables to
set like `EPOCHS` or `DATASET_NAME`, add additional `--env` args to your docker run command.

> Note that proxy vars are required when running on the Intel network in order
> to download the pretrained model from the model hub and datasets, if a dataset
> catalog is being used.
```
MODEL_NAME=efficientnet_b0
OUTPUT_DIR=<path to an output directory>
DATASET_DIR=<path to your dataset>

mkdir -p ${OUTPUT_DIR}

docker run --rm \
    -u $(id -u):$(id -g) \
    --env http_proxy=${http_proxy} \
    --env https_proxy=${https_proxy} \
    --env MODEL_NAME=${MODEL_NAME} \
    --env OUTPUT_DIR=${OUTPUT_DIR} \
    --env DATASET_DIR=${DATASET_DIR} \
    -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
    -v ${DATASET_DIR}:${DATASET_DIR} \
    -it intel/tlt:tensorflow
```

To run using a dataset from a catalog (like TFDS or torchvision), specify the
`DATASET_NAME`. The `DATASET_DIR` needs to be a writeable directory where the dataset
will be downloaded (the directory can be empty). Subsequent runs will reload
the dataset from the same directory. The command below shows an example of how
to run `efficientnet_b0` using PyTorch and the CIFAR10 dataset.
```
MODEL_NAME=efficientnet_b0
DATASET_NAME=CIFAR10
OUTPUT_DIR=<path to an output directory>
DATASET_DIR=<path to a dataset directory>

mkdir -p ${OUTPUT_DIR}
mkdir -p ${DATASET_DIR}

docker run --rm \
    -u $(id -u):$(id -g) \
    --env http_proxy=${http_proxy} \
    --env https_proxy=${https_proxy} \
    --env MODEL_NAME=${MODEL_NAME} \
    --env DATASET_NAME=${DATASET_NAME} \
    --env OUTPUT_DIR=${OUTPUT_DIR} \
    --env DATASET_DIR=${DATASET_DIR} \
    -v ${OUTPUT_DIR}:${OUTPUT_DIR} \
    -v ${DATASET_DIR}:${DATASET_DIR} \
    -it intel/tlt:pytorch
```

### Evaluation

After training, the model is exported to your output directory. The container
output prints a path to the saved model directory (since the model is saved to
a numbered subdirectory like `${OUTPUT_DIR}/efficientnet_b0/8`.

The following snippet shows how to evaluate a trained model using a dataset on
your local machine:
```
MODEL_DIR=<path to the numbered saved model directory>
DATASET_DIR=<path to your dataset>

docker run --rm \
    -u $(id -u):$(id -g) \
    -v ${MODEL_DIR}:${MODEL_DIR} \
    -v ${DATASET_DIR}:${DATASET_DIR} \
    -it intel/tlt:tensorflow \
    tlt eval --model-dir ${MODEL_DIR} --dataset-dir ${DATASET_DIR}
```

### Running unit tests

Once you've built the unit test container by setting `TEST_CONTAINER=True` when
running the `build.sh` you will have a test container like `intel/tlt:tensorflow-tests`
for TensorFlow or `intel/tlt:pytorch-tests` for PyTorch. The default command for
these container will run unit tests that have been marked for that framework.

The unit tests use datasets out of the `/tmp/data` directory and output is written to
the `/tmp/output` directory . If you want to utilize datasets that have been previously
downloaded, you can mount a volume for your dataset folder to `/tmp/data`. The command
below is an example of mounting a data directory and running pytorch tests.
```
DATASET_DIR=<path to your dataset directory>
OUTPUT_DIR=<path to a writeable output directory>

mkdir -p ${OUTPUT_DIR}

docker run --rm \
    --env http_proxy=${http_proxy} \
    --env https_proxy=${https_proxy} \
    -v ${DATASET_DIR}:/tmp/data \
    -v ${OUTPUT_DIR}:/tmp/output \
    -it intel/tlt:pytorch-tests
```

## Lint the dockerfiles

Use [hadolint](https://github.com/hadolint/hadolint) to check the dockerfile
best practices.

Run `hadolint` using docker:
```
# Navigate to the directory where the dockefiles are located
cd tlt/tools/docker/dockerfiles

# Pipe the `tf.Dockerfile` to `docker run`:
docker run --rm -i hadolint/hadolint < tf.Dockerfile

# Pipe the `pyt.Dockerfile` to `docker run`:
docker run --rm -i hadolint/hadolint < pyt.Dockerfile
```
