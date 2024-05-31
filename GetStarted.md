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

| Resource         | Minimum  |
| ---------------------------- | ---------- |
| CPU Cores | 8  (16+ recommended) |
| RAM | 16 GB (24-32+ GB recommended) |
| Disk space | 10 GB minimum (can vary based on datasets downloaded) |

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

   Use the Basic Installation instructions unless you plan on making code changes or installing the latest code from the repository.
   Please note that mixing basic and advanced installation options within the same virtual environment is not supported.

   a. **Basic Installation**

      ```
      pip install intel-transfer-learning-tool
      ```

   b. **Advanced Installation**

      Clone the repo:

      ```
      git clone https://github.com/Intel/transfer-learning.git
      cd transfer-learning
      ```

      Then either do an editable install to avoid a rebuild and
      install after each code change (preferred):

      ```
      pip install .
      ```

      or build and install a wheel:

      ```
      python setup.py bdist_wheel
      pip install dist/intel_transfer_learning_tool-0.7.0-py3-none-any.whl
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

## &#9314; Run the Intel Transfer Learning Tool

With the Intel Transfer Learning Tool, you can train AI models with TensorFlow or
PyTorch using either no-code CLI commands at a bash prompt, or low-code API
calls from a Python script. Both approaches provide the same opportunities for
training, evaluation, optimization, and benchmarking. With the CLI, no
programming experience is required, and you'll need basic Python knowledge to
use the API. Choose the approach that works best for you.


### a) Run Using the No-Code CLI

Let's continue from the previous step where you prepared the dataset, and train
a model using CLI commands.  This example uses the CLI to train an image
classifier to identify different types of flowers. You can see a list of all
available image classifier models using the command:

```
tlt list models --use-case image_classification
```

**Train a Model**

In this example, we'll use the `tlt train` command to retrain the PyTorch
efficientnet_b0 model using a food101 dataset from the
[PyTorch Datasets](https://pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html).
The `--dataset-dir` and `--output-dir` paths need to point to writable folders on your system.
```

tlt train -f pytorch --model-name efficientnet_b0 --dataset-name Food101 --dataset-dir "/tmp/data-${USER}" --output-dir "/tmp/output-${USER}"
```
```
Model name: efficientnet_b0
Framework: pytorch
Dataset name: Food101
Training epochs: 1
Dataset dir: /tmp/data-user
Output directory: /tmp/output-user

...
Epoch 1/1
----------
100%|██████████████████████████████████████████████████| 1776/1776 [27:02<00:00,  1.09it/s]                                       
Performing Evaluation
100%|██████████████████████████████████████████████████| 592/592 [08:33<00:00,  1.15it/s]                                         
Loss: 2.7038 - Acc: 0.3854 - Val Loss: 2.1242 - Val Acc: 0.4880
Training complete in 35m 37s
Saved model directory: /tmp/output-user/efficientnet_b0/1
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

### b) Run Using the Low-Code API

The following Python code example trains an image classification model with the PyTorch
RenderedSST2 dataset using API calls from Python.  The model is
benchmarked and quantized to INT8 precision for improved inference performance.

You can run the API example using a Jupyter notebook. See the [notebook setup
instructions](/notebooks/setup.md) for more details for preparing the Jupyter
notebook environment.

```python
import os
from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.types import FrameworkType, UseCaseType

username = os.getenv('USER', 'user')

# Specify a writable directory for the dataset to be downloaded
dataset_dir = '/tmp/data-{}'.format(username)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Specify a writeable directory for output (such as saved model files)
output_dir = '/tmp/output-{}'.format(username)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the model
model = model_factory.get_model(model_name="efficientnet_b0", framework=FrameworkType.PYTORCH)

# Download and preprocess the RenderedSST2 dataset from the torchvision datasets catalog
dataset = dataset_factory.get_dataset(dataset_dir=dataset_dir,
                                      dataset_name='RenderedSST2',
                                      use_case=UseCaseType.IMAGE_CLASSIFICATION,
                                      framework=FrameworkType.PYTORCH,
                                      dataset_catalog='torchvision')
dataset.preprocess(image_size=model.image_size, batch_size=32)
dataset.shuffle_split(train_pct=.75, val_pct=.25)

# Train the model using the dataset
model.train(dataset, output_dir=output_dir, epochs=1, ipex_optimize=False)

# Visualize the trained model result
import matplotlib.pyplot as plt
import numpy as np
images, labels = dataset.get_batch()

# Predict with a single batch
predictions = model.predict(images)

# Map the predicted ids to the class names
predictions = [dataset.class_names[id] for id in predictions]

# Display the results
plt.figure(figsize=(16,16))
plt.subplots_adjust(hspace=0.5)
for n in range(min(batch_size, 30)):
    plt.subplot(6,5,n+1)
    inp = images[n]
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    correct_prediction = labels[n] == predictions[n]
    color = "darkgreen" if correct_prediction else "crimson"
    title = predictions[n].title() if correct_prediction else "{}\n({})".format(predictions[n], labels[n]) 
    plt.title(title, fontsize=14, color=color)
    plt.axis('off')
_ = plt.suptitle("Model predictions", fontsize=16)
plt.show()
print("Correct predictions are shown in green")
print("Incorrect predictions are shown in red with the actual label in parenthesis")

# Export the model
saved_model_dir = model.export(output_dir=output_dir)

# Quantize the trained model
quantization_output = os.path.join(output_dir, "quantized_model")
model.quantize(quantization_output, dataset, overwrite_model=True)

# Benchmark the trained model using the Intel Neural Compressor config file
model.benchmark(dataset, saved_model_dir=quantization_output)
```

For more information on the API, see the [API Documentation](/api.md).

## Summary and Next Steps

The Intel Transfer Learning Tool can be used to develop an AI model and export
an Intel-optimized saved model for deployment. The sample CLI and API commands
we've presented show how to execute end-to-end transfer learning workflows. 

For the no-code CLI, you can follow a
complete example that includes training, evaluation, benchmarking, and quantization
in the datasets, as well as some additional models in the [Beyond Get Started
CLI example](examples/cli/README.md) documentation. You can also read about all the
CLI commands in the [CLI reference](/cli.md).

For the low-code API, read about the API in the [API Documentation](/api.md).

Find more CLI and API examples in our list of [Examples](examples/README.md).
