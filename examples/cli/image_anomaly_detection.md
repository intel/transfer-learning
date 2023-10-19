# Image Anomaly Detection Intel® Transfer Learning Tool CLI Example

## Transfer Learning Using CutPaste Feature Extraction and your Own Dataset

The example below shows how the Intel Transfer Learning Tool CLI can be used for image anomaly detection transfer
learning using your own dataset. The dataset is expected to be organized with subfolders for the "good" (non-defective)
images and any number of "bad" (or defective) classes. Or, the category folders can be arranged in subfolders named
"train" and "test". Either way, each subfolder should contain .jpg images for the class.

This example assumes you have downloaded the [MVTec dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad),
which has images of hazelnuts belonging to 5 classes: crack, cut, good, hole, and print. The extracted
dataset is already formatted in the expected format with subfolders for each class.
```bash
# Set dataset and output directories
export DATASET_DIR=/tmp/data/mvtec/hazelnut
export OUTPUT_DIR=/tmp/output
mkdir -p ${OUTPUT_DIR}

# Fine-tune a resnet50 feature extractor for anomaly detection using CutPaste and the hazelnut photos directory
tlt train \
    -f pytorch \
    --model-name resnet50 \
    --use-case image_anomaly_detection \
    --dataset-dir ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --epochs 2 \
    --cutpaste

# Evaluate the model exported after training
# Note that your --model-dir path may vary, since each training run creates a new directory
tlt eval \
    --model-dir /tmp/output/resnet50/1 \
    --dataset-dir ${DATASET_DIR}
```

## Transfer Learning Using SimSiam Feature Extraction and your Own Dataset

This example shows the Intel Transfer Learning Tool CLI being used for image anomaly detection feature extraction
and fine-tuning using the SimSiam method with manually downloaded weights and the bottle subset from 
[MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad).
```bash
# Set dataset and output directories
export DATASET_DIR=/tmp/data/mvtec/bottle
export OUTPUT_DIR=/tmp/output
mkdir -p ${OUTPUT_DIR}

# Download the starting checkpoints for the SimSiam feature extractor
wget https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar -P ${OUTPUT_DIR}

# Fine-tune a resnet18 feature extractor for anomaly detection using SimSiam and the bottle photos directory
tlt train \
    -f pytorch \
    --model-name resnet18 \
    --use-case image_anomaly_detection \
    --dataset-dir ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --epochs 2 \
    --init-checkpoints ${OUTPUT_DIR}/checkpoint_0099.pth.tar \
    --simsiam

# Evaluate the model exported after training
# Note that your --model-dir path may vary, since each training run creates a new directory
tlt eval \
    --model-dir ${OUTPUT_DIR}/resnet18/1 \
    --dataset-dir ${DATASET_DIR}
```

## Citations

Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: International Journal of Computer Vision 129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
