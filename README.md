# Transfer Learning Kit (TLK)

> Note that the `tlk` (Transfer Learning Kit) tool name is a placeholder until
> we have the actual tool name picked out.


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
```
> tlk list models --use-case image_classification
------------------------------
IMAGE CLASSIFICATION
------------------------------
densenet121 (pytorch)
densenet161 (pytorch)
efficientnet_b0 (pytorch)
efficientnet_b0 (tensorflow)
efficientnet_b1 (pytorch)
efficientnet_b1 (tensorflow)
efficientnet_b2 (pytorch)
efficientnet_b2 (tensorflow)
efficientnet_b3 (tensorflow)
efficientnet_b4 (tensorflow)
efficientnet_b5 (tensorflow)
efficientnet_b6 (tensorflow)
efficientnet_b7 (tensorflow)
efficientnetv2-b0 (tensorflow)
efficientnetv2-b1 (tensorflow)
efficientnetv2-b2 (tensorflow)
efficientnetv2-b3 (tensorflow)
efficientnetv2-s (tensorflow)
googlenet (pytorch)
inception_v3 (tensorflow)
mobilenet_v2 (pytorch)
mobilenet_v2_100_224 (tensorflow)
nasnet_large (tensorflow)
resnet18 (pytorch)
resnet50 (pytorch)
resnet_v1_50 (tensorflow)
resnet_v2_101 (tensorflow)
resnet_v2_50 (tensorflow)
shufflenet_v2_x1_0 (pytorch)
```

Train a model:
```
> tlk train -f tensorflow --model-name efficientnet_b0 --dataset-dir /tmp/data --output-dir /tmp/output --dataset-name tf_flowers
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
```
> tlk eval --model-dir /tmp/output/efficientnet_b0/1 --dataset-name tf_flowers --dataset-dir /tmp/data
Model directory: /tmp/output/efficientnet_b0/1
Dataset directory: /tmp/data
Dataset name: tf_flowers
Model name: efficientnet_b0
Loading model object for efficientnet_b0 using tensorflow
Loading saved model from: /tmp/output/efficientnet_b0/1/saved_model.pb
2022-05-13 12:47:19.809448: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-05-13 12:47:19.816455: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting:
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
Using dataset catalog 'tf_datasets', since no dataset catalog was specified
28/28 [==============================] - 8s 219ms/step - loss: 0.4080 - acc: 0.8996
```

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
tf_flowers.preprocess(image_size=efficientnet_b0_model.image_size, batch_size=32)
tf_flowers.shuffle_split(train_pct=.75, val_pct=.25)

# Train the model using the dataset
efficientnet_b0_model.train(tf_flowers, output_dir="/tmp/output", epochs=1)

# Evaluate the trained model
metrics = efficientnet_b0_model.evaluate(tf_flowers)
for metric_name, metric_value in zip(efficientnet_b0_model._model.metrics_names, metrics):
    print("{}: {}".format(metric_name, metric_value))

# Export the model
efficientnet_b0_model.export(output_dir="/tmp/output")
```
