# Image Classification Intel® Transfer Learning Tool CLI Example

## Transfer Learning Using your Own Dataset

The example below shows how the Intel Transfer Learning Tool CLI can be used for image classification transfer learning
using your own dataset. The dataset is expected to be organized with subfolders for each image
class. Each subfolder should contain .jpg images for the class. The name of the subfolder will
be used as the class label.

This example downloads a flower photos dataset from TensorFlow, which has images of
flowers belonging to 5 classes: daisy, dandelion, roses, sunflowers, and tulips. The extracted
dataset is already formatted in the expected format with subfolders for each class.
```bash
# Create dataset and output directories
DATASET_DIR=/tmp/data
OUTPUT_DIR=/tmp/output
mkdir -p ${DATASET_DIR}
mkdir -p ${OUTPUT_DIR}

# Download and extract the dataset
wget -P ${DATASET_DIR} https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
tar -xzf ${DATASET_DIR}/flower_photos.tgz -C ${DATASET_DIR}

# Set the DATASET_DIR to the extracted images folder
DATASET_DIR=${DATASET_DIR}/flower_photos

# Train resnet_v1_50 using the flower photos directory
tlt train \
    -f tensorflow \
    --model-name resnet_v1_50 \
    --dataset-dir ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --epochs 2

# Evaluate the model exported after training
# Note that your --model-dir path may vary, since each training run creates a new directory
tlt eval \
    --model-dir /tmp/output/resnet_v1_50/1 \
    --dataset-dir ${DATASET_DIR}
```

## Transfer Learning Using a Dataset from the TFDS Catalog

This example shows the Intel Transfer Learning Tool CLI being used for image classification transfer learning
using the `tf_flowers` dataset from the
[TensorFlow Datasets (TFDS) catalog](https://www.tensorflow.org/datasets/catalog/overview).

```bash
# Create dataset and output directories
DATASET_DIR=/tmp/data
OUTPUT_DIR=/tmp/output
mkdir -p ${DATASET_DIR}
mkdir -p ${OUTPUT_DIR}

# Name of the dataset to use
DATASET_NAME=tf_flowers

# Train resnet_v1_50 using the TFDS catalog dataset
tlt train \
    -f tensorflow \
    --model-name resnet_v1_50 \
    --dataset-name ${DATASET_NAME} \
    --dataset-dir ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --epochs 2

# Evaluate the model exported after training
# Note that your --model-dir path may vary, since each training run creates a new directory
tlt eval \
    --model-dir ${OUTPUT_DIR}/resnet_v1_50/1 \
    --dataset-name ${DATASET_NAME} \
    --dataset-dir ${DATASET_DIR}
```

## Citations

```
@ONLINE {tfflowers,
author = "The TensorFlow Team",
title = "Flowers",
month = "jan",
year = "2019",
url = "http://download.tensorflow.org/example_images/flower_photos.tgz" }
```
