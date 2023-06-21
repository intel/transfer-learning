# Get Started

This is a guide for getting started with Intel® Transfer Learning Tool and will
walk you through the steps to check system requirements, install, and then run
the tool with a couple of examples showing no-code CLI and low-code API
approaches.

<p align="center"><b>Intel Transfer Learning Tool Get Started Flow</b></p>

<img alt="Intel Transfer Learning Tool Get Started Flow" title="Intel Transfer Learning Tool Get Started Flow" src="images/TLT-GSG_flow.svg" width="800">

## &#9312; Check System Requirements

| Recommended Hardware         | Precision  |
| ---------------------------- | ---------- |
| Intel® 4th Gen Xeon® Scalable Performance processors | BF16 |
| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors | FP32 |

| Required Software         |
| ------------------------- |
| Linux\* system (validated on Ubuntu\* 20.04/22.04 LTS) |
| Python (3.8, 3.9, or 3.10) |
| Pip |
| Conda or Python virtualenv |
| git (only required for advanced installation) |

## &#9313; Install

1. **Install Dependencies**

   Install required packages using:

   ```
   sudo apt-get install build-essential python3-dev libgl1 libglib2.0-0
   ```

2. **Create and activate a Python3 virtual environment**

   We encourage you to use a Python virtual environment (virtualenv or conda)
   for consistent package management.  There are two ways to do this:

   a. Use `virtualenv`:

      ```
      virtualenv -p python3 tlt_dev_venv
      source tlt_dev_venv/bin/activate
      ```

   b. Or use `conda`:

      ```
      conda create --name tlt_dev_venv python=3.9
      conda activate tlt_dev_venv
      ```

3. **Install Intel Transfer Learning Tool**

   Use the Basic Installation instructions unless you plan on making code changes.

   a. **Basic Installation**

      ```
      pip install intel-transfer-learning-tool
      ```

   b. **Advanced Installation**

      Clone the repo:

      ```
      git clone https://github.com/IntelAI/transfer-learning.git
      cd transfer-learning
      ```

      Then either do an editable install to avoid a rebuild and
      install after each code change (preferred):

      ```
      pip install --editable .
      ```

      or build and install a wheel:

      ```
      python setup.py bdist_wheel
      pip install dist/intel_transfer_learning_tool-0.5.0-py3-none-any.whl
      ```


4. **Additional Feature-Specific Steps**

   * For distributed/multinode training, follow these additional
     [distributed training instructions](tlt/distributed/README.md).

5. **Verify Installation**

   Verify that your installation was successful by using the following
   command, which displays help information about the Intel Transfer Learning Tool:

   ```
   tlt --help
   ```

6. **Prepare the Dataset**

   The Intel Transfer Learning Tool can use datasets from existing dataset catalogs
   or custom datasets that you have on your machine.  The following CLI and API
   examples use the Intel Transfer Learning Tool's custom dataset option
   (`--dataset-dir`) with the TensorFlow flowers dataset.

   ```
   # Create a directory for the dataset to be downloaded
   DATASET_DIR=/tmp/dataset
   mkdir -p ${DATASET_DIR}

   # Download and extract the dataset (be sure https_proxy is set if needed)
   wget -P ${DATASET_DIR} https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
   tar -xzf ${DATASET_DIR}/flower_photos.tgz -C ${DATASET_DIR}

   # Set the DATASET_DIR to the extracted images folder
   DATASET_DIR=${DATASET_DIR}/flower_photos
   ```

   At this point, you should have a `flower_photos` folder with
   subfolders for `daisy`, `dandelion`, `roses`, `sunflower`, and `tulips`.

## &#9314; Run the Intel Transfer Learning Tool

With the Intel Transfer Learning Tool, you can train AI models with TensorFlow or
PyTorch using either no-code CLI commands at a bash prompt, or low-code API
calls from a Python script. Both approaches provide the same opportunities for
training, evaluation, optimization, and benchmarking. With the CLI, no
programming experience is required, and you'll need basic Python knowledge to
use the API. Choose the approach that works best for you.


### Run Using the No-Code CLI

Let's continue from the previous step where you prepared the dataset, and train
a model using CLI commands.  This example uses the CLI to train an image
classifier to identify different types of flowers. You can see a list of all
available image classifier models using the command:

```
tlt list models --use-case image_classification
```

**Train a Model**

In this example, we'll use the ``tlt train`` command to use the TensorFlow
ResNet50v1.5 model using the flowers dataset we already prepared and write the
trained model to a folder specified with `--output-dir`.

```
tlt train -f tensorflow --model-name resnet_v1_50 --dataset-dir ${DATASET_DIR} --output-dir /tmp/output
```
```
Model name: resnet_v1_50
Framework: tensorflow
Training epochs: 1
Dataset dir: /tmp/dataset/flower_photos
Output directory: /tmp/output
Found 3670 files belonging to 5 classes.
...
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
keras_layer (KerasLayer)    (None, 2048)              23561152
dense (Dense)               (None, 5)                 10245
=================================================================
Total params: 23,571,397
Trainable params: 10,245
Non-trainable params: 23,561,152
_________________________________________________________________
Checkpoint directory: /tmp/output/resnet_v1_50_checkpoints
86/86 [==============================] - 24s 248ms/step - loss: 0.4600 - acc: 0.8438
Saved model directory: /tmp/output/resnet_v1_50/1
```

After training completes, the `tlt train` command evaluates the model. The loss and
accuracy values are printed toward the end of the console output. The model is
exported to the output directory you specified in a numbered folder created for
each training run.

**Next Steps**

That ends this Get Started CLI example. As a next step, you can also follow the
[Beyond Get Started CLI Example](examples/cli/README.md) for a complete example
that includes evaluation, benchmarking, and quantization in the datasets.

Read about all the CLI commands in the [CLI reference](/cli.md).
Find more examples in our list of [Examples](examples/README.md).

### Run Using the Low-Code API

The following Python code example trains an image classification model with the TensorFlow
flowers dataset using API calls from Python.  The model is
benchmarked and quantized to INT8 precision for improved inference performance.

You can run the API example using a Jupyter notebook. See the [notebook setup
instructions](/notebooks/setup.md) for more details for preparing the Jupyter
notebook environment.

```python
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.types import FrameworkType, UseCaseType

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

# Quantize the trained model
quantization_output = os.path.join(output_dir, "quantized_model")
model.quantize(quantization_output, dataset, overwrite_model=True)

# Benchmark the trained model using the Intel Neural Compressor config file
model.benchmark(dataset, saved_model_dir=quantization_output)

# Do graph optimization on the trained model
optimization_output = os.path.join(output_dir, "optimized_model")
model.optimize_graph(optimization_output, overwrite_model=True)
```

For more information on the API, see the [API Documentation](/api.md).

## Summary and Next Steps

The Intel Transfer Learning Tool can be used to develop an AI model and export
an Intel-optimized saved model for deployment. The sample CLI and API commands
we've presented show how to execute end-to-end transfer learning workflows. 

For the no-code CLI, you can follow a
complete example that includes trainng, evaluation, benchmarking, and quantization
in the datasets, as well as some additional models in the [Beyond Get Started
CLI example](examples/cli/README.md) documentation. You can also read about all the
CLI commands in the [CLI reference](/cli.md).

For the low-code API, read about the API in the [API Documentation](/api.md).

Find more CLI and API examples in our list of [Examples](examples/README.md).
