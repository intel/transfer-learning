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

![alt text](https://raw.githubusercontent.com/IntelAI/transfer-learning/main/docs/images/features.png "TLT CLI and API")

## Get Started

### Requirements
* Linux* system (validated on Ubuntu* 20.04/22.04 LTS)
* Python3* (3.8, 3.9, or 3.10), Pip and Conda/Virtualenv
* Install required packages with `apt-get install build-essential python3-dev libgl1 libglib2.0-0`
* git (only required for the "Developer Installation")

### Create and activate a Python3 virtual environment
We encourage you to use a python virtual environment (virtualenv or conda) for consistent package management.
There are two ways to do this:

a. Using `virtualenv`:
   ```
   virtualenv -p python3 tlt_dev_venv
   source tlt_dev_venv/bin/activate
   ```

b. Or `conda`:
   ```
   conda create --name tlt_dev_venv python=3.9
   conda activate tlt_dev_venv
   ```

### Basic Installation
```
pip install intel-transfer-learning-tool
```

### Developer Installation
Use these instructions to install the Intel Transfer Learning Tool using a clone of the
GitHub repository. This can be done instead of the basic pip install, if you plan
on making code changes.

1. Clone this repo and navigate to the repo directory:
   ```
   git clone https://github.com/IntelAI/transfer-learning.git

   cd transfer-learning
   ```

2. Install the Intel Transfer Learning Tool by either:

   a. Building and installing the wheel:
      ```
      python setup.py bdist_wheel
      pip install dist/intel_transfer_learning_tool-0.4.0-py3-none-any.whl
      ```

   b. Or, do an editable install to avoid having to rebuild and install after each code change:
      ```
      pip install --editable .
      ```

### Additional Feature-Specific Steps:
 1. For TensorFlow text classification, this tensorflow-text Python library is also required:
    ```
    pip install tensorflow-text==2.11.0
    ```

 1. For distributed/multinode training, follow these additional [distributed training instructions](https://github.com/IntelAI/transfer-learning-tool/tree/main/tlt/distributed).

### Verify Installation

Verify that your installation was successful by using the following
command, which displays help information about the Intel Transfer Learning Tool:
```
tlt --help
```

### Prepare the Dataset

The Intel Transfer Learning Tool can use datasets from dataset catalogs or custom datasets that you have on your machine.

The following CLI and API examples use the custom dataset option (`--dataset-dir`) with the TensorFlow flowers dataset.
Prior to running these examples, download the flowers dataset from
[https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
and extract the files to a folder on your machine. After extracting the dataset,
you should have a `flower_photos` folder with subfolders for `daisy`, `dandelion`,
`roses`, `sunflower`, and `tulips`.

## Use the No-code CLI
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
...
```

See the [full list of supported models](https://github.com/IntelAI/transfer-learning-tool/blob/main/Models.md).

**Train a model**:
This example uses the CLI to train an image classifier to identify different types of flowers.
Make sure to specify your own file paths for `dataset-dir` and `output-dir`. The `dataset-dir` should
point to the [extracted flowers dataset](#prepare-the-dataset). For more information on using different
datasets, see the [CLI examples](https://github.com/IntelAI/transfer-learning-tool/tree/main/examples/cli).
```
tlt train -f tensorflow --model-name resnet_v1_50 --dataset-dir /tmp/dataset/flower_photos --output-dir /tmp/output
```
```
Model name: resnet_v1_50
Framework: tensorflow
Dataset name: tf_flowers
Training epochs: 1
Dataset dir: /tmp/dataset/flower_photos
Output directory: /tmp/output
Found 3670 files belonging to 5 classes.
...
Saved model directory: /tmp/output/resnet_v1_50/1
```

After training completes, the model is exported to the output directory specified in your command. The actual directory name
is printed out to the console. A numbered folder is created for each training run.

The training command also evalutes the trained model and prints out accuracy and loss metrics.
Evaluation can also be called separately using `tlt eval`. The trained model can also be benchmarked
using `tlt benchmark` or quantized using `tlt quantize`.
See the [CLI documentation](https://github.com/IntelAI/transfer-learning-tool/blob/main/examples/cli/README.md) for more examples using the CLI.

## Use the Low-code API
The following example trains an image classification model with the TensorFlow flowers dataset using the API.
Additionally, the model is benchmarked and quantized to int8 precision for improved inference performance.
If you want to run the API using a Jupyter notebook, see the [notebook setup instructions](https://github.com/IntelAI/transfer-learning-tool/blob/main/notebooks/setup.md).

```python
from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.types import FrameworkType, UseCaseType
import os

# Specify the directory where the TensorFlow flowers dataset has been downloaded and extracted
# (https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
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

For more information on the API see: [https://intelai.github.io/transfer-learning](https://intelai.github.io/transfer-learning).

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

Check out these Reference Kits and Workflows that use Intel Transfer Learning Tool:

* [Breast Cancer Detection](https://github.com/IntelAI/transfer-learning/tree/main/workflows/disease_prediction)
* [Anomaly Detection](https://github.com/IntelAI/transfer-learning/tree/main/workflows/vision_anomaly_detection)

## Support

The Intel Transfer Learning Tool team tracks bugs and enhancement requests using
[GitHub issues](https://github.com/IntelAI/transfer-learning-tool/issues). Before submitting a
suggestion or bug report, search the existing GitHub issues to see if your issue has already been reported.

*Other names and brands may be claimed as the property of others. [Trademarks](http://www.intel.com/content/www/us/en/legal/trademarks.html)

#### DISCLAIMER: ####
These scripts are not intended for benchmarking Intel platforms. For any performance and/or benchmarking information on specific Intel platforms, visit https://www.intel.ai/blog.

Intel is committed to the respect of human rights and avoiding complicity in human rights abuses, a policy reflected in the Intel Global Human Rights Principles. Accordingly, by accessing the Intel material on this platform you agree that you will not use the material in a product or application that causes or contributes to a violation of an internationally recognized human right.

#### License: ####
Intel® Transfer Learning Tool is licensed under Apache License Version 2.0.

#### Datasets: ####
To the extent that any public datasets are referenced by Intel or accessed using tools or code on this site those datasets are provided by the third party indicated as the data source. Intel does not create the data, or datasets, and does not warrant their accuracy or quality. By accessing the public dataset(s) you agree to the terms associated with those datasets and that your use complies with the applicable license. [DATASETS](https://github.com/IntelAI/transfer-learning-tool/blob/main/DATASETS.md)

Intel expressly disclaims the accuracy, adequacy, or completeness of any public datasets, and is not liable for any errors, omissions, or defects in the data, or for any reliance on the data.  Intel is not liable for any liability or damages relating to your use of public datasets.
