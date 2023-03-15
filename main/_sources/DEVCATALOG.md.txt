# Intel® Transfer Learning Tool

Transfer learning workflows use the knowledge learned by a pre-trained model on a large dataset to improve the
performance of a related problem with a smaller dataset. Intel® Transfer Learning Tool makes transfer learning
workflows easier and faster across a variety of AI use cases. This open-source Python* library leverages public
pretrained model hubs, Intel-optimized deep learning frameworks, and your custom dataset to efficiently generate new
models optimized for Intel hardware.
 
This document provides information, links, and instructions for the Intel Transfer Learning Tool as well as Jupyter*
notebooks and examples that demonstrate its usage.

## Overview

The Intel Transfer Learning Tool offers both a low-code API and a no-code CLI for training AI models with TensorFlow*
and PyTorch*.

Features:
* PyTorch and TensorFlow support
* Over 100 image classification and text classification models from Torchvision, TensorFlow datasets, and Hugging Face
* Automatically create a trainable classification layer customized for your dataset
* Bring your own dataset or get started quickly with built-in datasets
* Dataset scaling, cropping, batching, and splitting
* APIs for prediction, evaluation, and benchmarking
* Export model for deployment or resume training from checkpoints

Intel Optimizations:
* Boost performance with Intel® Optimization for TensorFlow and Intel® Extension for PyTorch
* Quantize to INT8 to reduce model size and speed up inference using Intel® Neural Compressor
* Optimize model for FP32 inference using Intel Neural Compressor
* Reduce training time with auto-mixed precision for select hardware platforms
* Further reduce training time with multinode training for PyTorch

For more details, visit the [Intel Transfer Learning Tool](https://github.com/IntelAI/transfer-learning-tool) 
GitHub repository.

## Hardware Requirements

| Recommended Hardware         | Precision  |
| ---------------------------- | ---------- |
| Intel® 4th Gen Xeon® Scalable Performance processors | BF16 |
| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors | FP32 |

## How it Works
Run simple CLI commands at a bash prompt or make API calls in a Python* script to: 

1. Download public pretrained models, replace the classification layer, and train them using your own dataset
2. Minimize training time with Intel-optimized frameworks and low-precision features from Intel Neural Compressor 
3. Export a saved model optimized for inference on Intel CPUs

![alt text](images/features.png "TLT CLI and API")

## Get Started

### Requirements
1. Linux* system (verified on Ubuntu* 20.04), CPU-only
2. Python3* (3.8, 3.9, or 3.10), Pip/Conda and Virtualenv
3. Install required packages with `apt-get install build-essential python3-dev libgl1 libglib2.0-0`

### Install

```
pip install intel-transfer-learning-tool
```

### Use the CLI
Use `tlt --help` to see the list of CLI commands. More detailed information on each
command can be found using `tlt <command> --help` (like `tlt train --help`).

List the available models:
```
tlt list models --use-case image_classification
```
```
------------------------------
IMAGE CLASSIFICATION
------------------------------
alexnet (pytorch)
convnext_base (pytorch)
convnext_large (pytorch)
convnext_small (pytorch)
convnext_tiny (pytorch)
densenet121 (pytorch)
densenet161 (pytorch)
densenet169 (pytorch)
densenet201 (pytorch)
efficientnet_b0 (pytorch)
efficientnet_b0 (tensorflow)
efficientnet_b1 (pytorch)
efficientnet_b1 (tensorflow)
efficientnet_b2 (pytorch)
efficientnet_b2 (tensorflow)
efficientnet_b3 (pytorch)
efficientnet_b3 (tensorflow)
efficientnet_b4 (pytorch)
efficientnet_b4 (tensorflow)
efficientnet_b5 (pytorch)
efficientnet_b5 (tensorflow)
efficientnet_b6 (pytorch)
efficientnet_b6 (tensorflow)
efficientnet_b7 (pytorch)
efficientnet_b7 (tensorflow)
efficientnetv2-b0 (tensorflow)
efficientnetv2-b1 (tensorflow)
efficientnetv2-b2 (tensorflow)
efficientnetv2-b3 (tensorflow)
efficientnetv2-s (tensorflow)
googlenet (pytorch)
inception_v3 (tensorflow)
mnasnet0_5 (pytorch)
mnasnet1_0 (pytorch)
mobilenet_v2 (pytorch)
mobilenet_v2_100_224 (tensorflow)
mobilenet_v3_large (pytorch)
mobilenet_v3_small (pytorch)
nasnet_large (tensorflow)
regnet_x_16gf (pytorch)
regnet_x_1_6gf (pytorch)
regnet_x_32gf (pytorch)
regnet_x_3_2gf (pytorch)
regnet_x_400mf (pytorch)
regnet_x_800mf (pytorch)
regnet_x_8gf (pytorch)
regnet_y_16gf (pytorch)
regnet_y_1_6gf (pytorch)
regnet_y_32gf (pytorch)
regnet_y_3_2gf (pytorch)
regnet_y_400mf (pytorch)
regnet_y_800mf (pytorch)
regnet_y_8gf (pytorch)
resnet101 (pytorch)
resnet152 (pytorch)
resnet18 (pytorch)
resnet34 (pytorch)
resnet50 (pytorch)
resnet_v1_50 (tensorflow)
resnet_v2_101 (tensorflow)
resnet_v2_50 (tensorflow)
resnext101_32x8d (pytorch)
resnext50_32x4d (pytorch)
shufflenet_v2_x0_5 (pytorch)
shufflenet_v2_x1_0 (pytorch)
vgg11 (pytorch)
vgg11_bn (pytorch)
vgg13 (pytorch)
vgg13_bn (pytorch)
vgg16 (pytorch)
vgg16_bn (pytorch)
vgg19 (pytorch)
vgg19_bn (pytorch)
vit_b_16 (pytorch)
vit_b_32 (pytorch)
vit_l_16 (pytorch)
vit_l_32 (pytorch)
wide_resnet101_2 (pytorch)
wide_resnet50_2 (pytorch)
```

**Train a model**:
Make sure to specify your own file paths for `dataset-dir` and `output-dir`
```
tlt train -f tensorflow --model-name resnet_v1_50 --dataset-dir /tmp/dataset/flower_photos --output-dir /tmp/output
```
```
Model name: resnet_v1_50
Framework: tensorflow
Training epochs: 1
Dataset dir: /tmp/dataset/flower_photos
Output directory: /tmp/output
Found 3670 files belonging to 5 classes.
...
```

### Use the API
This example runs the same workload as above using the API instead of the CLI. Additionally, the model is benchmarked and
quantized to int8 precision for improved inference performance.

```python
from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.types import FrameworkType, UseCaseType
import os

# Specify a directory for the dataset to be downloaded
dataset_dir = os.environ["DATASET_DIR"] if "DATASET_DIR" in os.environ else \
    os.path.join(os.environ["HOME"], "dataset")

# Specify a directory for output
output_dir = os.environ["OUTPUT_DIR"] if "OUTPUT_DIR" in os.environ else \
    os.path.join(os.environ["HOME"], "output")

# Get the model
model = model_factory.get_model(model_name="resnet_v1_50", framework=FrameworkType.TENSORFLOW)

# Load and preprocess a dataset
dataset = dataset_factory.load_dataset(dataset_dir = os.path.join(dataset_dir, "flower_photos"),
                                       use_case=UseCaseType.IMAGE_CLASSIFICATION, \
                                       framework=FrameworkType.TENSORFLOW)
dataset.preprocess(image_size=model.image_size, batch_size=32)
dataset.shuffle_split(train_pct=.75, val_pct=.25)

# Train the model using the dataset
model.train(dataset, output_dir=output_dir, epochs=1)

# Evaluate the trained model
metrics = model.evaluate(dataset)
for metric_name, metric_value in zip(model._model.metrics_names, metrics):
    print("{}: {}".format(metric_name, metric_value))

# Export the model
saved_model_dir = model.export(output_dir=output_dir)

# Create an Intel Neural Compressor config file
inc_config_file = os.path.join(output_dir, "inc_config.yaml")
model.write_inc_config_file(inc_config_file, dataset=dataset, batch_size=512, overwrite=True,
                            accuracy_criterion_relative=0.01, exit_policy_timeout=0,
                            exit_policy_max_trials=10, tuning_workspace=os.path.join(output_dir, "nc_workspace"))

# Quantize the trained model
quantization_output = os.path.join(output_dir, "quantized_model")
model.quantize(saved_model_dir, quantization_output, inc_config_file)

# Benchmark the trained model using the Intel Neural Compressor config file
model.benchmark(quantization_output, inc_config_file, 'performance')

# Do graph optimization on the trained model
optimization_output = os.path.join(output_dir, "optimized_model")
model.optimize_graph(saved_model_dir, optimization_output)
```

For more details, visit the [Intel Transfer Learning Tool](https://github.com/IntelAI/transfer-learning-tool) 
GitHub repository.

## Summary and Next Steps

You have just learned how Intel Transfer Learning Tool can be used to quickly develop an AI model and export
an Intel-optimized saved model for deployment. With the sample CLI and API commands above, you have executed simple
end-to-end transfer learning workflows. For more details, check out the tutorial Jupyter*
notebooks, and for real-world examples check out the reference workflows.

### Tutorial Jupyter* Notebooks

| Notebook | Use Case | Framework| Description |
| ---------| ---------|----------|-------------|
| [Text Classification with TensorFlow using the Intel® Transfer Learning Tool](https://github.com/IntelAI/transfer-learning-tool/tree/main/notebooks/text_classification/tlt_api_tf_text_classification) | Text Classification | TensorFlow and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to fine tune a BERT model from TF Hub using binary text classification datasets. |
| [Text Classification with Pytorch using the Intel® Transfer Learning Tool](https://github.com/IntelAI/transfer-learning-tool/tree/main/notebooks/text_classification/tlt_api_pyt_text_classification) | Text Classification | PyTorch and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to fine tune a BERT model from Huggingface using binary text classification datasets. |
| [Image Classification with TensorFlow using Intel® Transfer Learning Tool](https://github.com/IntelAI/transfer-learning-tool/tree/main/notebooks/image_classification/tlt_api_tf_image_classification) | Image Classification | TensorFlow and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to do transfer learning for image classification using a TensorFlow model. |
| [Image Classification with PyTorch using Intel® Transfer Learning Tool](https://github.com/IntelAI/transfer-learning-tool/tree/main/notebooks/image_classification/tlt_api_pyt_image_classification) | Image Classification | PyTorch and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to do transfer learning for image classification using a PyTorch model. |

### Examples

Coming soon: AI Reference Kits that use Intel Transfer Learning Tool

## Support

The Intel Transfer Learning Tool team tracks bugs and enhancement requests using 
[GitHub issues](https://github.com/IntelAI/transfer-learning-tool/issues). Before submitting a
suggestion or bug report, search the existing GitHub issues to see if your issue has already been reported.

*Other names and brands may be claimed as the property of others. [Trademarks](http://www.intel.com/content/www/us/en/legal/trademarks.html)
