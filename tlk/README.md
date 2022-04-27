# Transfer Learning Kit (TLK

## Build and Install

Requirements:
* Linux system (or WSL2 on Windows)
* git
* python3

1. Clone this repo and navigate to the repo directory:
   ```
   git clone git@github.com:intel-innersource/frameworks.ai.transfer-learning.git
   cd frameworks.ai.transfer-learning
   ```

1. Create and activate a Python3 virtual environment using `virtualenv`:
   ```
   python3 -m virtualenv tlk_env
   source tlk_env/bin/activate
   ```

   Or `conda`:
   ```
   conda create --name tlk_env python=3.8
   conda activate tlk_env
   ```

1. Install the tool with the `tensorflow` and/or `pytorch` option by either building
   and installing the wheel:
   ```
   python setup.py bdist_wheel --universal
   pip install dist/tlk-0.0.1-py2.py3-none-any.whl[tensorflow]
   ```
   Or for developers, do an editable install:
   ```
   pip install --editable .[tensorflow]
   ```

## Run the CLI

Use `tlk --help` to see the list of CLI commands. More detailed information on each
command can be found using `tlk <command> --help` (like `tlk train --help`).

List the available models:
```bash
> tlk list models
Image Classification
----------------------------------------
resnet_v1_50 (tensorflow)
resnet_v2_50 (tensorflow)
resnet_v2_101 (tensorflow)
mobilenet_v2_100_224 (tensorflow)
efficientnetv2-s (tensorflow)
efficientnet_b0 (tensorflow)
efficientnet_b1 (tensorflow)
efficientnet_b2 (tensorflow)
inception_v3 (tensorflow)
nasnet_large (tensorflow)
```

Train a model:
```bash
> tlk train -f tensorflow --model-name efficientnet_b0 --dataset-dir /tmp/data --output-dir /tmp/output --dataset-name tf_flowers --dataset-catalog tf_datasets
Model name: efficientnet_b0
Framework: tensorflow
Dataset name: tf_flowers
Dataset catalog: tf_datasets
Dataset dir: /tmp/data
Output directory: /tmp/output
2022-04-27 10:19:12.651375: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-04-27 10:19:12.658987: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 keras_layer (KerasLayer)    (None, 1280)              4049564

 dense (Dense)               (None, 5)                 6405

=================================================================
Total params: 4,055,969
Trainable params: 6,405
Non-trainable params: 4,049,564
_________________________________________________________________
None
86/86 [==============================] - 22s 215ms/step - loss: 0.5106 - acc: 0.8438
2022-04-27 10:19:38.545222: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
Saved model directory: /tmp/output/efficientnet_b0/1
```

Evaluate a trained model:
> Not implemented yet

## Use the API

```python
from tlk.datasets import dataset_factory
from tlk.models import model_factory
from tlk.utils.types import FrameworkType, UseCaseType

# Get the model
efficientnet_b0_model = model_factory.get_model(model_name="efficientnet_b0", \
                                                framework=FrameworkType.TENSORFLOW)

# Get and preprocess a dataset
tf_flowers = dataset_factory.get_dataset(dataset_dir="/tmp/data",
                                         use_case=UseCaseType.IMAGE_CLASSIFICATION, \
                                         framework=FrameworkType.TENSORFLOW, \
                                         dataset_name="tf_flowers", \
                                         dataset_catalog="tf_datasets")
tf_flowers.preprocess(image_size=efficientnet_b0_model.image_size,
                      batch_size=32)

# Train the model using the dataset
efficientnet_b0_model.train(tf_flowers, epochs=1)

# Export the model
efficientnet_b0_model.export(output_dir="/tmp/output")
```
