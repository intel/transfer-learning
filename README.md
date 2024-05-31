*Note: You may find it easier to read about Intel Transfer Learning tool, follow the Get
Started guide, and browse the API material from our published documentation site
https://intel.github.io/transfer-learning.*

<!-- SkipBadges -->

# Intel® Transfer Learning Tool

Transfer learning workflows use the knowledge learned by a pre-trained model on
a large dataset to improve the performance of a related problem with a smaller
dataset.

## What is Intel® Transfer Learning Tool

Intel® Transfer Learning Tool makes it easier and faster for you to
create transfer learning workflows across a variety of AI use cases. Its
open-source Python\* library leverages public pretrained model hubs,
Intel-optimized deep learning frameworks, and your custom dataset to efficiently
generate new models optimized for Intel hardware.

This project documentation provides information, resource links, and instructions for the Intel
Transfer Learning Tool as well as Jupyter\* notebooks and examples that
demonstrate its usage.

**Features:**
* Supports PyTorch\* and TensorFlow\*
* Select from over [100 computer vision and natural language processing models](Models.md) from
  Torchvision, PyTorch Hub, TensorFlow Hub, Keras, and Hugging Face
* Use cases include:
  * Image Classification
  * Text Classification
  * Text Generation
  * Vision Anomaly Detection
* Use your own custom dataset or get started quickly with built-in datasets
* Automatically create a trainable classification layer customized for your dataset
* Pre-process your dataset using scaling, cropping, batching, and splitting
* Use APIs for prediction, evaluation, and benchmarking
* Export your model for deployment or resume training from checkpoints

**Intel Optimizations:**
* Boost performance with Intel® Optimization for TensorFlow and Intel® Extension for PyTorch
* Quantize to INT8 to reduce model size and speed up inference using Intel® Neural Compressor
* Optimize model for FP32 inference using Intel Neural Compressor
* Reduce training time with auto-mixed precision for select hardware platforms
* Further reduce training time with multinode training

## How the Intel Transfer Learning Tool Works

The Intel Transfer Learning Tool lets you train AI models with TensorFlow or
PyTorch using either no-code command line interface (CLI) commands at a bash
prompt, or low-code application programming interface (API) calls from a Python
script.

Use your own dataset or select an existing image or text dataset listed in the
[public datasets](DATASETS.md) documentation. Construct your own CLI or API commands for training, evaluation,
and optimization using the TensorFlow or PyTorch framework, and finally export
your saved model optimized for inference on Intel CPUs.

An overview of the Intel Transfer Learning Tool flow is shown in this
figure:

<p align="center"><b>Intel Transfer Learning Tool Flow</b></p>

<img alt="Intel Transfer Learning Tool Flow" title="Intel Transfer Learing Tool Flow" src="images/TLT-tool_flow.svg" width="600">

## Get Started

Check out the [Get Started Guide](GetStarted.md) which will walk you through the
steps to check system requirements, install, and then run the tool with a couple of
examples showing no-code CLI and low-code API approaches. After that, you can check out
these additional CLI and API [Examples](examples/README.md).

<!-- ExpandGetStarted-Start -->
As described in the [Get Started Guide](GetStarted.md), once you have a Python 
environment set up, you do a basic install of the Intel Transfer Learning
Tool. Note that the default installation includes PyTorch and not TensorFlow.
Here are some examples of commands you will find in the [Get Started Guide](GetStarted.md):

```
pip install intel-transfer-learning-tool
```

Then you can use the Transfer Learning Tool CLI interface (tlt) to train a
PyTorch image classification model (RenderedSST2), download and use an
existing built-in dataset (tf_flowers), and save the trained model to
`/tmp/output` using this one command:

```
tlt train --framework pytorch --model-name efficientnet_b0 --dataset-name RenderedSST2 \
   --output-dir /tmp/output --dataset-dir /tmp/data
```

TensorFlow can be used by installing with the following command:

```
pip install intel-transfer-learning-tool[tensorflow]
```

Use `tlt --help` to see the list of CLI commands.  More detailed help for each
command can be found using, for example, `tlt train --help`.

<!-- ExpandGetStarted-End -->

## Note on Evaluation and Bias

Intel Transfer Learning Tool provides standard evaluation metrics such as accuracy and loss for validation/test/train sets. While important, it's essential to acknowledge that these metrics may not explicitly capture biases. Users should be cautious and consider potential biases by analyzing disparities in the data and model prediction. Techniques such as confusion matrices, PR curves, ROC curves, local attribution-based and `gradCAM` explanations, can all be good indicators for bias. Clear documentation of model behavior and performance is also crucial for iterative bias mitigation. [Intel® Explainable AI Tools](https://github.com/Intel/intel-xai-tools/tree/main) provides components that demonstrate the aformentioned techniques with [Explainer](https://github.com/Intel/intel-xai-tools/tree/main/explainer), a simple API providing post-hoc model distillation and visualization methods, as well as The [Model Card Generator](https://github.com/Intel/intel-xai-tools/tree/main/model_card_gen) which provides an interactive HTML report that containing these workflows and demonstrations of model behavior.

## Support

The Intel Transfer Learning Tool team tracks bugs and enhancement requests using
[GitHub issues](https://github.com/Intel/transfer-learning-tool/issues). Before submitting a
suggestion or bug report, search the existing GitHub issues to see if your issue has already been reported.

See [Legal Information](Legal.md) for Disclaimers, Trademark, and Licensing information.
